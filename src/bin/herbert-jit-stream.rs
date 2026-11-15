//! herbert-jit-stream — the memory bandwidth ceiling, STREAM's four kernels emitted with iced-x86.
//!
//! Copy (`c = a`), Scale (`b = s·c`), Add (`c = a + b`), Triad (`a = b + s·c`, one FMA), and a
//! read-only reduction (`Σ a`, four accumulators: the decode's pattern, which reads the weights),
//! on `f64` arrays, in AVX-512 (zmm) and AVX2 (ymm), with normal stores (`vmovapd`) or non-temporal
//! ones (`vmovntpd` then `sfence`). Every kernel is emitted here (no intrinsics, no compiler
//! between the instruction and the measure); four vectors per iteration.
//!
//! The arrays (1 GiB each by default, `--size-mib`) are mapped anonymous, with `MADV_HUGEPAGE`
//! unless `--no-thp`, and first touched by the emitted fill kernel on the threads that will read
//! them, each its own contiguous slice of whole pages. Threads pinned through the engine's pool
//! (`ThreadPool::new_on_core_ids`): physical cores first, in the engine's order (CCDs interleaved),
//! then their SMT siblings for the points beyond (marked `smt`). The calling thread is worker 0 and
//! is pinned to the first CPU.
//!
//! Per point, `--reps` repetitions (10); best, median and worst in GB/s (10⁹ bytes/s). Bytes
//! counted the STREAM way: Copy and Scale 2 accesses per element, Add and Triad 3, the read 1;
//! `--write-allocate` adds one access per store for the normal stores (the line read before it is
//! written), not for the non-temporal ones. The results are checked at the end of each series, as
//! STREAM does. One readable line and one JSON line (`{"stream":...}`) per point.
//!
//! `herbert-jit-stream [--size-mib 1024] [--threads 1,2,4,6,8,12,16] [--reps 10] [--isa
//! avx512,avx2]  [--stores normal,nt] [--write-allocate] [--no-thp]`

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() {
    if let Err(e) = imp::main() {
        eprintln!("herbert-jit-stream: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    eprintln!("herbert-jit-stream: x86-64 Linux only (the kernels are emitted with iced-x86, the threads pinned by sched_setaffinity)");
    std::process::exit(1);
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod imp {
    use herbert_jit_q4::pool::thread_pool::ThreadPool;
    use iced_x86::IcedError;
    use std::time::Instant;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Kind {
        Copy,
        Scale,
        Add,
        Triad,
        Read,
        Fill,
    }

    impl Kind {
        fn name(self) -> &'static str {
            match self {
                Kind::Copy => "Copy",
                Kind::Scale => "Scale",
                Kind::Add => "Add",
                Kind::Triad => "Triad",
                Kind::Read => "Read",
                Kind::Fill => "Fill",
            }
        }
        /// Arrays read and written, the STREAM count.
        fn accesses(self) -> (usize, usize) {
            match self {
                Kind::Copy | Kind::Scale => (1, 1),
                Kind::Add | Kind::Triad => (2, 1),
                Kind::Read => (1, 0),
                Kind::Fill => (0, 1),
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Isa {
        Avx512,
        Avx2,
    }

    impl Isa {
        fn name(self) -> &'static str {
            match self {
                Isa::Avx512 => "avx512",
                Isa::Avx2 => "avx2",
            }
        }
    }

    /// `extern "sysv64" fn(a, b, c, bytes, &s, out)`: every kernel takes the six, each uses its
    /// own. `bytes` is a multiple of four vectors.
    pub type KernelFn =
        unsafe extern "sysv64" fn(*mut f64, *mut f64, *mut f64, usize, *const f64, *mut f64);

    /// The kernel of one register width: `$r` the four data registers, `$s` the scalar, `$ptr` the
    /// memory operand of that width, `$zero` the integer xor of that width (AVX-512F / AVX2).
    macro_rules! body {
        ($asm:ident, $kind:expr, $nt:expr, $v:expr, [$r0:ident, $r1:ident, $r2:ident, $r3:ident], $s:ident, $ptr:ident, $zero:ident) => {{
            let regs = [$r0, $r1, $r2, $r3];
            let mut top = $asm.create_label();
            let mut done = $asm.create_label();
            $asm.test(rcx, rcx)?;
            $asm.jz(done)?;
            if matches!($kind, Kind::Scale | Kind::Triad | Kind::Fill) {
                $asm.vbroadcastsd($s, qword_ptr(r8))?;
            }
            if $kind == Kind::Read {
                for r in regs {
                    $asm.$zero(r, r, r)?;
                }
            }
            $asm.xor(eax, eax)?;
            $asm.set_label(&mut top)?;
            for (u, r) in regs.into_iter().enumerate() {
                let o = (u * $v) as i32;
                // The store: `dst` the destination array.
                let store =
                    |asm: &mut CodeAssembler, dst: AsmRegister64, r| -> Result<(), IcedError> {
                        if $nt {
                            asm.vmovntpd($ptr(dst + rax + o), r)
                        } else {
                            asm.vmovapd($ptr(dst + rax + o), r)
                        }
                    };
                match $kind {
                    Kind::Copy => {
                        $asm.vmovapd(r, $ptr(rdi + rax + o))?;
                        store(&mut $asm, rdx, r)?;
                    }
                    Kind::Scale => {
                        $asm.vmulpd(r, $s, $ptr(rdx + rax + o))?;
                        store(&mut $asm, rsi, r)?;
                    }
                    Kind::Add => {
                        $asm.vmovapd(r, $ptr(rdi + rax + o))?;
                        $asm.vaddpd(r, r, $ptr(rsi + rax + o))?;
                        store(&mut $asm, rdx, r)?;
                    }
                    Kind::Triad => {
                        $asm.vmovapd(r, $ptr(rsi + rax + o))?;
                        $asm.vfmadd231pd(r, $s, $ptr(rdx + rax + o))?;
                        store(&mut $asm, rdi, r)?;
                    }
                    Kind::Read => {
                        $asm.vaddpd(r, r, $ptr(rdi + rax + o))?;
                    }
                    Kind::Fill => {
                        store(&mut $asm, rdi, $s)?;
                    }
                }
            }
            $asm.add(rax, (4 * $v) as i32)?;
            $asm.cmp(rax, rcx)?;
            $asm.jb(top)?;
            $asm.set_label(&mut done)?;
            if $nt {
                $asm.sfence()?;
            }
            if $kind == Kind::Read {
                $asm.vaddpd($r0, $r0, $r1)?;
                $asm.vaddpd($r2, $r2, $r3)?;
                $asm.vaddpd($r0, $r0, $r2)?;
                $asm.vmovupd($ptr(r9), $r0)?;
            }
            $asm.vzeroupper()?;
            $asm.ret()?;
        }};
    }

    pub fn emit(kind: Kind, isa: Isa, nt: bool) -> Result<Vec<u8>, IcedError> {
        use iced_x86::code_asm::*;
        let mut a = CodeAssembler::new(64)?;
        match isa {
            Isa::Avx512 => body!(
                a,
                kind,
                nt,
                64usize,
                [zmm0, zmm1, zmm2, zmm3],
                zmm7,
                zmmword_ptr,
                vpxorq
            ),
            Isa::Avx2 => body!(
                a,
                kind,
                nt,
                32usize,
                [ymm0, ymm1, ymm2, ymm3],
                ymm7,
                ymmword_ptr,
                vpxor
            ),
        }
        a.assemble(0)
    }

    /// One emitted kernel on its own executable page.
    pub struct Kernel {
        _map: memmap2::Mmap,
        pub f: KernelFn,
    }

    pub fn compile(kind: Kind, isa: Isa, nt: bool) -> Result<Kernel, String> {
        let code = emit(kind, isa, nt).map_err(|e| format!("emit {kind:?} {isa:?}: {e}"))?;
        let mut map = memmap2::MmapMut::map_anon(code.len()).map_err(|e| e.to_string())?;
        map[..code.len()].copy_from_slice(&code);
        let map = map.make_exec().map_err(|e| e.to_string())?;
        // SAFETY: the page holds the code just emitted for this signature.
        let f = unsafe { std::mem::transmute::<*const u8, KernelFn>(map.as_ptr()) };
        Ok(Kernel { _map: map, f })
    }

    /// An anonymous mapping of `bytes`, 2 MiB aligned, never touched here.
    struct Array {
        ptr: *mut f64,
        base: usize,
        len: usize,
    }

    impl Array {
        fn new(bytes: usize, thp: bool) -> Result<Array, String> {
            let huge = 2usize << 20;
            let len = bytes + huge;
            // SAFETY: an anonymous private mapping, released in `drop`.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(format!(
                    "mmap of {bytes} bytes: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let base = p as usize;
            let aligned = (base + huge - 1) & !(huge - 1);
            if thp {
                // SAFETY: inside the mapping.
                unsafe { libc::madvise(aligned as *mut libc::c_void, bytes, libc::MADV_HUGEPAGE) };
            }
            Ok(Array {
                ptr: aligned as *mut f64,
                base,
                len,
            })
        }
    }

    impl Drop for Array {
        fn drop(&mut self) {
            // SAFETY: the mapping made in `new`.
            unsafe { libc::munmap(self.base as *mut libc::c_void, self.len) };
        }
    }

    fn pin_self(cpu: usize) {
        // SAFETY: a cpu_set_t on the stack.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }

    /// The online logical CPUs: `/sys/devices/system/cpu/online` (`0-23`, `0-3,8-11`).
    fn online_cpus() -> Vec<usize> {
        let s = std::fs::read_to_string("/sys/devices/system/cpu/online").unwrap_or_default();
        let mut out = Vec::new();
        for part in s.trim().split(',').filter(|p| !p.is_empty()) {
            match part.split_once('-') {
                Some((a, b)) => {
                    if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                        out.extend(a..=b);
                    }
                }
                None => {
                    if let Ok(a) = part.parse() {
                        out.push(a);
                    }
                }
            }
        }
        out
    }

    struct Opts {
        size_mib: usize,
        threads: Vec<usize>,
        reps: usize,
        isas: Vec<Isa>,
        stores: Vec<bool>,
        write_allocate: bool,
        thp: bool,
    }

    fn parse() -> Result<Opts, String> {
        let avx512 = herbert_jit_q4::core::isa::feature("avx512f");
        let mut o = Opts {
            size_mib: 1024,
            threads: vec![1, 2, 4, 6, 8, 12, 16],
            reps: 10,
            isas: if avx512 {
                vec![Isa::Avx512, Isa::Avx2]
            } else {
                vec![Isa::Avx2]
            },
            stores: vec![false, true],
            write_allocate: false,
            thp: true,
        };
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{a}: value missing"));
            match a.as_str() {
                "--size-mib" => {
                    o.size_mib = value()?.parse().map_err(|e| format!("--size-mib: {e}"))?
                }
                "--threads" => {
                    o.threads = value()?
                        .split(',')
                        .map(|t| t.trim().parse().map_err(|e| format!("--threads: {e}")))
                        .collect::<Result<_, _>>()?
                }
                "--reps" => o.reps = value()?.parse().map_err(|e| format!("--reps: {e}"))?,
                "--isa" => {
                    o.isas = value()?
                        .split(',')
                        .map(|s| match s.trim() {
                            "avx512" if avx512 => Ok(Isa::Avx512),
                            "avx512" => Err("--isa avx512: this CPU has no AVX-512F".to_string()),
                            "avx2" => Ok(Isa::Avx2),
                            x => Err(format!("--isa: {x}? (avx512, avx2)")),
                        })
                        .collect::<Result<_, _>>()?
                }
                "--stores" => {
                    o.stores = value()?
                        .split(',')
                        .map(|s| match s.trim() {
                            "normal" => Ok(false),
                            "nt" => Ok(true),
                            x => Err(format!("--stores: {x}? (normal, nt)")),
                        })
                        .collect::<Result<_, _>>()?
                }
                "--write-allocate" => o.write_allocate = true,
                "--no-thp" => o.thp = false,
                "-h" | "--help" => {
                    println!("herbert-jit-stream [--size-mib 1024] [--threads 1,2,4,6,8,12,16] [--reps 10] [--isa avx512,avx2] [--stores normal,nt] [--write-allocate] [--no-thp]");
                    std::process::exit(0);
                }
                x => return Err(format!("unknown argument {x}")),
            }
        }
        if o.reps == 0 || o.size_mib == 0 {
            return Err("--reps and --size-mib must be positive".into());
        }
        Ok(o)
    }

    #[derive(Clone, Copy)]
    struct Ptrs {
        a: usize,
        b: usize,
        c: usize,
        out: usize,
        chunk_bytes: usize,
    }

    /// One collective of `t` threads running `k` on their slices; its wall seconds.
    fn run(pool: &ThreadPool, t: usize, k: KernelFn, p: Ptrs, s: f64) -> Result<f64, String> {
        let s_addr = &s as *const f64 as usize;
        let t0 = Instant::now();
        pool.collective(
            t,
            |id| {
                let off = id * p.chunk_bytes;
                // SAFETY: each thread its own slice of whole pages inside the three arrays, its own
                // 64 bytes of `out`.
                unsafe {
                    k(
                        (p.a + off) as *mut f64,
                        (p.b + off) as *mut f64,
                        (p.c + off) as *mut f64,
                        p.chunk_bytes,
                        s_addr as *const f64,
                        (p.out + id * 64) as *mut f64,
                    )
                }
            },
            || {},
        )
        .map_err(|e| e.to_string())?;
        Ok(t0.elapsed().as_secs_f64())
    }

    pub fn main() -> Result<(), String> {
        let o = parse()?;
        let physical: Vec<usize> = herbert_jit_q4::pool::topology::physical_core_ids().to_vec();
        let mut cpus = physical.clone();
        for c in online_cpus() {
            if !cpus.contains(&c) {
                cpus.push(c);
            }
        }
        if cpus.is_empty() {
            return Err("no CPU list (topology)".into());
        }
        pin_self(cpus[0]);
        let bytes = o.size_mib << 20;
        println!(
            "herbert-jit-stream: 3 arrays of {} MiB (f64), {} repetitions, THP {}, CPU order [{}] ({} physical cores first)",
            o.size_mib,
            o.reps,
            if o.thp { "advised" } else { "off" },
            cpus.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
            physical.len()
        );
        println!(
            "bytes counted: STREAM convention, Copy/Scale 2 accesses per element, Add/Triad 3, Read 1{}; GB/s = 10^9 bytes/s",
            if o.write_allocate { ", plus 1 per store for the normal stores (write-allocate)" } else { "" }
        );
        let s = 3.0f64;
        let mut failures = 0usize;
        for &t in &o.threads {
            if t == 0 || t > cpus.len() {
                println!("threads {t}: skipped ({} logical CPUs)", cpus.len());
                continue;
            }
            let smt = t > physical.len();
            let pool = ThreadPool::new_on_core_ids(cpus[..t].to_vec());
            // Whole 4 KiB pages per thread (and so whole groups of four zmm).
            let chunk_bytes = (bytes / t) & !4095;
            let used = chunk_bytes * t;
            let n = used / 8;
            let (a, b, c) = (
                Array::new(bytes, o.thp)?,
                Array::new(bytes, o.thp)?,
                Array::new(bytes, o.thp)?,
            );
            let mut out = vec![0f64; t * 8 + 8];
            let p = Ptrs {
                a: a.ptr as usize,
                b: b.ptr as usize,
                c: c.ptr as usize,
                out: out.as_mut_ptr() as usize,
                chunk_bytes,
            };
            for &isa in &o.isas {
                let fill = compile(Kind::Fill, isa, false)?;
                for (series, &nt) in o.stores.iter().enumerate() {
                    // First touch (the first series) and reset, each thread on its own slice.
                    let init = |arr: usize, v: f64| run(&pool, t, fill.f, Ptrs { a: arr, ..p }, v);
                    init(p.a, 1.0)?;
                    init(p.b, 2.0)?;
                    init(p.c, 0.0)?;
                    let with_read = series == 0;
                    let mut kinds = vec![Kind::Copy, Kind::Scale, Kind::Add, Kind::Triad];
                    if with_read {
                        kinds.push(Kind::Read);
                    }
                    let ks: Vec<Kernel> = kinds
                        .iter()
                        .map(|&k| compile(k, isa, nt && k != Kind::Read))
                        .collect::<Result<_, _>>()?;
                    let mut times = vec![Vec::with_capacity(o.reps); kinds.len()];
                    let (mut aj, mut bj, mut cj) = (1.0f64, 2.0f64, 0.0f64);
                    let mut read_sum = 0.0f64;
                    for _ in 0..o.reps {
                        for (i, k) in ks.iter().enumerate() {
                            out.iter_mut().for_each(|x| *x = 0.0);
                            times[i].push(run(&pool, t, k.f, p, s)?);
                            if kinds[i] == Kind::Read {
                                read_sum = out.iter().sum();
                            }
                        }
                        cj = aj;
                        bj = s * cj;
                        cj = aj + bj;
                        aj = s.mul_add(cj, bj);
                    }
                    for (i, &k) in kinds.iter().enumerate() {
                        let (r, w) = k.accesses();
                        let wa = if o.write_allocate && !(nt && k != Kind::Read) {
                            w
                        } else {
                            0
                        };
                        let counted = (r + w + wa) * used;
                        let mut ts = times[i].clone();
                        ts.sort_by(|x, y| x.total_cmp(y));
                        let gbs = |sec: f64| counted as f64 / sec / 1e9;
                        let (best, med, worst) =
                            (gbs(ts[0]), gbs(ts[ts.len() / 2]), gbs(ts[ts.len() - 1]));
                        let store = if k == Kind::Read {
                            "-"
                        } else if nt {
                            "nt"
                        } else {
                            "normal"
                        };
                        println!(
                            "threads {t:>2}{} {:<6} {:<6} {:<5}  best {best:7.1}  median {med:7.1}  worst {worst:7.1} GB/s  ({:.2} GiB counted, best {:.2} ms)",
                            if smt { " smt" } else { "    " },
                            isa.name(),
                            store,
                            k.name(),
                            counted as f64 / (1u64 << 30) as f64,
                            ts[0] * 1e3
                        );
                        println!(
                            "{}",
                            serde_json::json!({"stream": {
                                "threads": t, "smt": smt, "cpus": &cpus[..t], "isa": isa.name(), "store": store, "kernel": k.name(),
                                "array_bytes": used, "counted_bytes": counted, "write_allocate": wa > 0, "reps": o.reps,
                                "best_gbs": best, "median_gbs": med, "worst_gbs": worst,
                                "times_s": times[i], "thp": o.thp,
                            }})
                        );
                    }
                    // STREAM's check: every element, then the read's sum.
                    let check = |arr: usize, want: f64, name: &str| -> usize {
                        // SAFETY: the `used` bytes of the array, written by the kernels above.
                        let v = unsafe { std::slice::from_raw_parts(arr as *const f64, n) };
                        let bad = v
                            .iter()
                            .filter(|&&x| ((x - want) / want).abs() > 1e-13)
                            .count();
                        if bad > 0 {
                            println!(
                                "CHECK FAILED: {name}: {bad} of {n} elements differ from {want}"
                            );
                        }
                        bad
                    };
                    let bad = check(p.a, aj, "a") + check(p.b, bj, "b") + check(p.c, cj, "c");
                    let read_bad =
                        with_read && ((read_sum - aj * n as f64) / (aj * n as f64)).abs() > 1e-9;
                    if read_bad {
                        println!(
                            "CHECK FAILED: Read: sum {read_sum} against {}",
                            aj * n as f64
                        );
                    }
                    if bad == 0 && !read_bad {
                        println!("threads {t:>2} {} {}: results checked (a = {aj}, b = {bj}, c = {cj}{})", isa.name(), if nt { "nt" } else { "normal" }, if with_read { ", Σa exact to 1e-9" } else { "" });
                    }
                    failures += bad + usize::from(read_bad);
                }
            }
            drop(pool);
        }
        if failures > 0 {
            return Err(format!("{failures} check failures"));
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every emitted kernel against the scalar loop, on 4 096 elements.
        #[test]
        fn kernels_match_the_scalar_loop() {
            let n = 4096;
            let isas: Vec<Isa> = if herbert_jit_q4::core::isa::feature("avx512f") {
                vec![Isa::Avx512, Isa::Avx2]
            } else {
                vec![Isa::Avx2]
            };
            for isa in isas {
                for nt in [false, true] {
                    let mut a: Vec<f64> = (0..n + 8).map(|i| i as f64 * 0.5).collect();
                    let mut b: Vec<f64> = (0..n + 8).map(|i| 1.0 + i as f64).collect();
                    let mut c: Vec<f64> = (0..n + 8).map(|i| 3.0 - i as f64 * 0.25).collect();
                    // 64-byte alignment: start at the first aligned element.
                    let al = |v: &mut Vec<f64>| {
                        v.as_mut_ptr()
                            .wrapping_add((64 - (v.as_ptr() as usize % 64)) % 64 / 8)
                    };
                    let (pa, pb, pc) = (al(&mut a), al(&mut b), al(&mut c));
                    let s = 3.0f64;
                    let mut out = [0f64; 8];
                    let ra: Vec<f64> = unsafe { std::slice::from_raw_parts(pa, n).to_vec() };
                    let rb: Vec<f64> = unsafe { std::slice::from_raw_parts(pb, n).to_vec() };
                    let rc: Vec<f64> = unsafe { std::slice::from_raw_parts(pc, n).to_vec() };
                    let call = |k: Kind, out: &mut [f64; 8]| {
                        let kern = compile(k, isa, nt && k != Kind::Read).unwrap();
                        unsafe { (kern.f)(pa, pb, pc, n * 8, &s, out.as_mut_ptr()) };
                    };
                    let get = |p: *mut f64| unsafe { std::slice::from_raw_parts(p, n).to_vec() };
                    call(Kind::Copy, &mut out);
                    assert_eq!(get(pc), ra, "{isa:?} nt {nt}: Copy");
                    call(Kind::Scale, &mut out);
                    let sb: Vec<f64> = ra.iter().map(|x| s * x).collect();
                    assert_eq!(get(pb), sb, "{isa:?} nt {nt}: Scale");
                    call(Kind::Add, &mut out);
                    let ac: Vec<f64> = ra.iter().zip(&sb).map(|(x, y)| x + y).collect();
                    assert_eq!(get(pc), ac, "{isa:?} nt {nt}: Add");
                    call(Kind::Triad, &mut out);
                    let ta: Vec<f64> = sb.iter().zip(&ac).map(|(b, c)| s.mul_add(*c, *b)).collect();
                    assert_eq!(get(pa), ta, "{isa:?} nt {nt}: Triad");
                    call(Kind::Read, &mut out);
                    let want: f64 = ta.iter().sum();
                    let got: f64 = out.iter().sum();
                    assert!(
                        ((got - want) / want).abs() < 1e-12,
                        "{isa:?}: Read {got} against {want}"
                    );
                    let fill = compile(Kind::Fill, isa, false).unwrap();
                    let v = 7.5f64;
                    unsafe { (fill.f)(pa, pb, pc, n * 8, &v, out.as_mut_ptr()) };
                    assert!(get(pa).iter().all(|&x| x == 7.5), "{isa:?}: Fill");
                    let _ = (&rb, &rc);
                }
            }
        }
    }
}
