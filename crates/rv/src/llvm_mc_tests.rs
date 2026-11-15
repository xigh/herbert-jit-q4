//! Every encoding of the crate against `llvm-mc`: a corpus of words (each scalar instruction the
//! assembler has, each vector load and store form, each operation of [`VOPS`]), decoded to text by
//! [`decode`]; `llvm_mc_golden.txt` holds, for each word, the text `llvm-mc` assembled back to the
//! same word. To check it again after a change:
//!
//! ```sh
//! cargo test -p rv print_corpus -- --ignored --nocapture --test-threads 1 | grep '^[0-9a-f]\{8\}' \
//!   > /tmp/corpus.txt
//! cut -f2 /tmp/corpus.txt | llvm-mc -triple=riscv64 -show-encoding \
//!   -mattr=+m,+a,+f,+d,+zfh,+zba,+zbb,+zbs,+zihintpause,+v,+zvfbfmin,+zvfbfwma,+zfbfmin
//! ```
//!
//! and compare the encodings with the first column.

use crate::vector::{Form, VOPS};
use crate::*;

/// The words of the corpus, in a fixed order.
fn corpus() -> Vec<u32> {
    let (a0, a1, a2, t0) = (x(10), x(11), x(12), x(5));
    let (f1, f2, f3, f4) = (f(1), f(2), f(3), f(4));
    let mut a = Asm::new();
    let back = a.here();
    macro_rules! each {
        ($($m:ident),* ; $args:tt) => { $( a.$m $args; )* };
    }
    each!(add, sub, sll, slt, sltu, xor, srl, sra, or, and, addw, subw, sllw, srlw, sraw, mul,
        mulh, mulhsu, mulhu, div, divu, rem, remu, mulw, divw, divuw, remw, remuw, sh1add, sh2add,
        sh3add, add_uw, sh1add_uw, sh2add_uw, sh3add_uw, andn, orn, xnor, min, minu, max, maxu,
        rol, ror, rolw, rorw; (a0, a1, a2));
    each!(addi, slti, sltiu, xori, ori, andi, addiw; (a0, a1, -7));
    each!(slli, srli, srai, slli_uw, rori, bseti, bclri, binvi, bexti; (a0, a1, 33));
    each!(slliw, srliw, sraiw, roriw; (a0, a1, 7));
    each!(clz, ctz, cpop, clzw, ctzw, cpopw, sext_b, sext_h, zext_h, rev8, orc_b, mv, not, neg,
        seqz, snez, sext_w, zext_w; (a0, a1));
    each!(lb, lh, lw, ld, lbu, lhu, lwu; (a0, a1, -16));
    each!(sb, sh, sw, sd; (a2, a1, 24));
    each!(amoswap_w, amoadd_w, amoxor_w, amoand_w, amoor_w, amomin_w, amomax_w, amominu_w,
        amomaxu_w, amoswap_d, amoadd_d, amoxor_d, amoand_d, amoor_d, amomin_d, amomax_d, amominu_d,
        amomaxu_d, sc_w, sc_d; (a0, a2, a1, Order::AcqRel));
    a.amoadd_d(a0, a2, a1, Order::Acquire);
    a.amoadd_d(a0, a2, a1, Order::Release);
    a.amoadd_d(a0, a2, a1, Order::Relaxed);
    a.lr_w(a0, a1, Order::Acquire);
    a.lr_d(a0, a1, Order::Relaxed);
    a.lui(a0, 0x12345);
    a.auipc(a0, 0x1);
    a.jalr(x(1), a1, 8);
    a.fence(Fence::R, Fence::RW);
    a.fence_tso();
    a.fence_i();
    a.pause();
    a.ecall();
    a.ebreak();
    a.csrrw(a0, Csr::FRM, a1);
    a.csrrs(a0, Csr::FFLAGS, a1);
    a.csrrc(a0, Csr::FCSR, a1);
    a.csrrwi(a0, Csr::FRM, 3);
    a.rdtime(a0);
    a.rdcycle(a0);
    each!(flw, fld, flh; (f1, a1, 8));
    each!(fsw, fsd, fsh; (f1, a1, -8));
    each!(fadd_s, fsub_s, fmul_s, fdiv_s, fadd_d, fsub_d, fmul_d, fdiv_d, fsgnj_s, fsgnjn_s,
        fsgnjx_s, fmin_s, fmax_s, fsgnj_d, fsgnjn_d, fsgnjx_d, fmin_d, fmax_d; (f1, f2, f3));
    each!(fmadd_s, fmsub_s, fnmsub_s, fnmadd_s, fmadd_d, fmsub_d, fnmsub_d, fnmadd_d; (f1, f2, f3,
        f4));
    each!(fsqrt_s, fsqrt_d, fmv_s, fneg_s, fabs_s, fmv_d, fcvt_d_s, fcvt_s_bf16; (f1, f2));
    each!(fclass_s, fmv_x_w, fmv_x_d, fmv_x_h; (a0, f2));
    each!(fmv_w_x, fmv_d_x, fmv_h_x, fcvt_d_w, fcvt_d_wu; (f1, a1));
    a.fcvt_s_d(f1, f2, Rm::Rne);
    a.fcvt_bf16_s(f1, f2, Rm::Rtz);
    a.beq(a0, a1, back);
    a.bltu(a0, a1, back);
    a.jal(x(1), back);
    a.li(a0, 0x1234_5678_9abc);
    for (sew, lmul) in [
        (Sew::E8, Lmul::Mf2),
        (Sew::E16, Lmul::M1),
        (Sew::E32, Lmul::M8),
        (Sew::E64, Lmul::M4),
    ] {
        a.vsetvli(t0, a0, Vtype::new(sew, lmul));
        a.vsetvli(t0, a0, Vtype::new(sew, lmul).tu().mu());
    }
    a.vsetivli(t0, 9, Vtype::new(Sew::E16, Lmul::M2).alt());
    a.vsetvl(t0, a0, a1);
    for sew in [Sew::E8, Sew::E16, Sew::E32, Sew::E64] {
        a.vle(sew, v(8), a1);
        a.vleff(sew, v(8), a1);
        a.vse(sew, v(8), a1);
        a.vlse(sew, v(8), a1, a2);
        a.vsse(sew, v(8), a1, a2);
        a.vluxei(sew, v(8), a1, v(16));
        a.vloxei(sew, v(8), a1, v(16));
        a.vsuxei(sew, v(8), a1, v(16));
        a.vsoxei(sew, v(8), a1, v(16));
        a.vlseg(3, sew, v(8), a1);
        a.vsseg(2, sew, v(8), a1);
        a.vlsseg(4, sew, v(8), a1, a2);
        a.vssseg(2, sew, v(8), a1, a2);
        a.vl_r(2, sew, v(8), a1);
    }
    a.vs_r(4, v(8), a1);
    a.vlm(v(8), a1);
    a.vsm(v(8), a1);
    let code = a.finish().expect("the corpus assembles");
    let mut words: Vec<u32> = code
        .bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Every operation of the table, `vd = v8`, `vs2 = v16`, the `vs1` / `rs1` / `fs1` / immediate
    // field 4 (or the sub-opcode), unmasked except the merges (`v0`).
    for op in VOPS {
        let (vs2, vs1, vm) = match op.form {
            Form::Unary(c) | Form::XUnary(c) | Form::Whole(c) => (16, c, 1),
            Form::XS | Form::FS => (16, 0, 1),
            Form::SX | Form::SF | Form::MvV | Form::MvI => (0, 4, 1),
            Form::Id => (0, 0b10001, 1),
            Form::MergeVv | Form::MergeVx | Form::MergeVi | Form::MergeVf => (16, 4, 0),
            _ => (16, 4, 1),
        };
        words.push(
            op.funct6 << 26 | vm << 25 | vs2 << 20 | vs1 << 15 | op.funct3 << 12 | 8 << 7 | 0x57,
        );
    }
    words
}

/// Each word of the corpus decodes to the text `llvm-mc` assembles back to it.
#[test]
fn words_match_llvm_mc() {
    let golden: std::collections::HashMap<u32, &str> = include_str!("llvm_mc_golden.txt")
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (w, t) = l.split_once('\t').expect("word<TAB>text");
            (u32::from_str_radix(w, 16).expect("a hexadecimal word"), t)
        })
        .collect();
    let words = corpus();
    for w in &words {
        let text = decode(*w).map(|i| i.text).unwrap_or_default();
        assert_eq!(
            golden.get(w).copied(),
            Some(text.as_str()),
            "{w:08x}: decoded as {text:?}"
        );
    }
    assert_eq!(golden.len(), words.len(), "the golden file has other words");
}

/// The corpus, one `word<TAB>text` a line, for `llvm-mc` (the module's header).
#[test]
#[ignore = "prints the corpus for llvm-mc"]
fn print_corpus() {
    // A line of its own: the test harness prints its name before the first line.
    println!();
    for w in corpus() {
        let text = decode(w).map(|i| i.text).unwrap_or_else(|| "?".into());
        println!("{w:08x}\t{text}");
    }
}
