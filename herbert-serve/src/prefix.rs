//! `--prefix-cache DIR`: the K/V of a request's stable prefix (the template's rendering of the
//! system messages and the tools alone, when its tokens begin the request's) kept on disk, so that
//! a restarted server does not compute it again.
//!
//! One file per prefix, `prefix-KEY.kv`: a header (the format, the description of the engine, the
//! positions, the bytes, a checksum), the prefix's tokens, then the K/V of positions `0..n` as the
//! engine's [`crate::Backend::kv_save`] gives them. KEY hashes the description and the tokens; a
//! file is used only when its description (executable, device, driver, model, tuning, capacity:
//! [`crate::Backend::kv_fingerprint`], the template and the options the worker adds) and its tokens
//! are the request's, its checksum right. Written to `.partial-PID-KEY` then renamed, by a thread
//! of its own (the request does not wait for the disk); a load cut short leaves no half file, the
//! partial files of other processes are removed at start. At most `files` files are kept, the least
//! recently used removed. An error is a warning: the prefix is then computed as without the option.

use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// FNV-1a, 64 bits.
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The first bytes of a prefix file: herbert-jit-q4's prefix cache (a file of another magic is not
/// read).
const MAGIC: &[u8; 8] = b"HJQPFX01";
/// Bumped whenever the file's layout or the meaning of its bytes changes.
const FORMAT: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Header {
    format: u32,
    desc: String,
    n: usize,
    kv_bytes: usize,
    checksum: String,
}

pub struct PrefixCache {
    dir: PathBuf,
    files: usize,
    desc: String,
    /// The tokens of the files of this description, read at start and added by [`Self::store`]
    /// ([`Self::best`]).
    index: std::sync::Mutex<Vec<(PathBuf, Vec<u32>)>>,
}

/// A fast checksum of the tokens and the K/V (64-bit words folded, the tail bytes after): it
/// catches a damaged file, not a forged one.
fn checksum(tokens: &[u32], kv: &[u8]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut fold = |bytes: &[u8]| {
        let (words, tail) = bytes.as_chunks::<8>();
        for w in words {
            h = (h ^ u64::from_le_bytes(*w))
                .wrapping_mul(0x0000_0100_0000_01b3)
                .rotate_left(29);
        }
        for &b in tail {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    fold(
        &tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    fold(kv);
    format!("{h:016x}")
}

fn token_bytes(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}

impl PrefixCache {
    pub fn new(dir: &Path, files: usize, desc: String) -> std::io::Result<PrefixCache> {
        std::fs::create_dir_all(dir)?;
        let pid = std::process::id();
        for e in std::fs::read_dir(dir)?.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with(".partial-") && !name.starts_with(&format!(".partial-{pid}-")) {
                let _ = std::fs::remove_file(e.path());
            }
        }
        let pc = PrefixCache {
            dir: dir.to_path_buf(),
            files: files.max(1),
            desc,
            index: Default::default(),
        };
        let index: Vec<(PathBuf, Vec<u32>)> = pc
            .listing()
            .into_iter()
            .filter_map(|(_, path)| pc.read_tokens(&path).ok().map(|t| (path, t)))
            .collect();
        *pc.index.lock().unwrap() = index;
        Ok(pc)
    }

    /// The tokens of a file of this description (its header and its tokens read, not its K/V).
    fn read_tokens(&self, path: &Path) -> Result<Vec<u32>, String> {
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let h = self.read_header(&mut f)?;
        let mut tb = vec![0u8; h.n * 4];
        f.read_exact(&mut tb).map_err(|e| e.to_string())?;
        Ok(tb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }

    fn read_header(&self, f: &mut std::fs::File) -> Result<Header, String> {
        let mut head = [0u8; 12];
        f.read_exact(&mut head).map_err(|e| e.to_string())?;
        if &head[..8] != MAGIC {
            return Err("not a prefix cache file".into());
        }
        let hl = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        if hl > 1 << 20 {
            return Err("header too long".into());
        }
        let mut hb = vec![0u8; hl];
        f.read_exact(&mut hb).map_err(|e| e.to_string())?;
        let h: Header = serde_json::from_slice(&hb).map_err(|e| e.to_string())?;
        if h.format != FORMAT {
            return Err(format!("format {} (this build reads {FORMAT})", h.format));
        }
        if h.desc != self.desc {
            return Err("written by another executable, device, model, tuning or capacity".into());
        }
        Ok(h)
    }

    /// A file holds exactly `tokens`.
    pub fn has(&self, tokens: &[u32]) -> bool {
        self.path(&self.key(tokens)).exists()
    }

    /// The most tokens a file holds that begin `ids` (all but its last token) and are more than
    /// `more`, when one does.
    pub fn best(&self, ids: &[u32], more: usize) -> Option<usize> {
        let limit = ids.len().saturating_sub(1);
        let index = self.index.lock().unwrap();
        index
            .iter()
            .filter(|(path, t)| {
                t.len() > more && t.len() <= limit && ids.starts_with(t) && path.exists()
            })
            .map(|(_, t)| t.len())
            .max()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The key of a prefix: the description and its tokens hashed.
    pub fn key(&self, tokens: &[u32]) -> String {
        let mut b = self.desc.as_bytes().to_vec();
        b.push(0);
        b.extend(token_bytes(tokens));
        format!("{:016x}", fnv64(&b))
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("prefix-{key}.kv"))
    }

    /// The tokens and the K/V of a file, when its description is this cache's and its checksum
    /// right.
    fn read(&self, path: &Path) -> Result<(Vec<u32>, Vec<u8>), String> {
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let h = self.read_header(&mut f)?;
        let mut tb = vec![0u8; h.n * 4];
        f.read_exact(&mut tb).map_err(|e| e.to_string())?;
        let tokens: Vec<u32> = tb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let mut kv = vec![0u8; h.kv_bytes];
        f.read_exact(&mut kv).map_err(|e| e.to_string())?;
        if f.read(&mut [0u8; 1]).map_err(|e| e.to_string())? != 0 {
            return Err("bytes after the K/V".into());
        }
        if checksum(&tokens, &kv) != h.checksum {
            return Err("checksum mismatch".into());
        }
        Ok((tokens, kv))
    }

    /// The K/V of `tokens` when a file holds them (its time of use then now).
    pub fn load(&self, tokens: &[u32]) -> Option<Vec<u8>> {
        let path = self.path(&self.key(tokens));
        if !path.exists() {
            return None;
        }
        match self.read(&path) {
            Ok((t, kv)) if t == tokens => {
                touch(&path);
                Some(kv)
            }
            Ok(_) => {
                eprintln!(
                    "[prefix-cache] {}: other tokens under the same key: ignored",
                    path.display()
                );
                None
            }
            Err(e) => {
                eprintln!("[prefix-cache] {}: {e}: ignored", path.display());
                None
            }
        }
    }

    /// Mark the file of `tokens`, if any, as used now: a prefix the K/V held already is the one
    /// loaded at the next start when it was the last used.
    pub fn used(&self, tokens: &[u32]) {
        touch(&self.path(&self.key(tokens)));
    }

    /// The most recently used file of this description: its tokens, its K/V and its path.
    pub fn newest(&self) -> Option<(Vec<u32>, Vec<u8>, PathBuf)> {
        for (_, path) in self.listing() {
            match self.read(&path) {
                Ok((t, kv)) => {
                    touch(&path);
                    return Some((t, kv, path));
                }
                Err(e) => eprintln!("[prefix-cache] {}: {e}: not loaded", path.display()),
            }
        }
        None
    }

    /// The cache's files, the most recently used first.
    fn listing(&self) -> Vec<(SystemTime, PathBuf)> {
        let mut v: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&self.dir)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| {
                        let n = e.file_name().to_string_lossy().to_string();
                        n.starts_with("prefix-") && n.ends_with(".kv")
                    })
                    .map(|e| {
                        (
                            e.metadata()
                                .and_then(|m| m.modified())
                                .unwrap_or(SystemTime::UNIX_EPOCH),
                            e.path(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        v.sort_by_key(|e| std::cmp::Reverse(e.0));
        v
    }

    /// Write the file of `tokens` on a thread of its own, then keep the `files` most recently used.
    pub fn store(&self, tokens: Vec<u32>, kv: Vec<u8>) -> std::thread::JoinHandle<()> {
        let key = self.key(&tokens);
        let path = self.path(&key);
        {
            let mut index = self.index.lock().unwrap();
            index.retain(|(p, _)| *p != path);
            index.push((path.clone(), tokens.clone()));
        }
        let tmp = self
            .dir
            .join(format!(".partial-{}-{key}", std::process::id()));
        let header = Header {
            format: FORMAT,
            desc: self.desc.clone(),
            n: tokens.len(),
            kv_bytes: kv.len(),
            checksum: checksum(&tokens, &kv),
        };
        let (dir, files) = (self.dir.clone(), self.files);
        let desc = self.desc.clone();
        std::thread::spawn(move || {
            let t = std::time::Instant::now();
            let res = (|| -> std::io::Result<()> {
                use std::io::Write;
                let hb = serde_json::to_vec(&header).map_err(std::io::Error::other)?;
                let mut f =
                    std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp)?);
                f.write_all(MAGIC)?;
                f.write_all(&(hb.len() as u32).to_le_bytes())?;
                f.write_all(&hb)?;
                f.write_all(&token_bytes(&tokens))?;
                f.write_all(&kv)?;
                let f = f.into_inner().map_err(|e| e.into_error())?;
                f.sync_all()?;
                drop(f);
                std::fs::rename(&tmp, &path)
            })();
            match res {
                Ok(()) => {
                    eprintln!(
                        "[prefix-cache] {} written: {} tokens, {:.1} MiB, in {:.2} s",
                        path.display(),
                        tokens.len(),
                        kv.len() as f64 / (1u64 << 20) as f64,
                        t.elapsed().as_secs_f64()
                    );
                    let pc = PrefixCache {
                        dir,
                        files,
                        desc,
                        index: Default::default(),
                    };
                    for (_, old) in pc.listing().into_iter().skip(files) {
                        match std::fs::remove_file(&old) {
                            Ok(()) => eprintln!(
                                "[prefix-cache] {} removed (more than {files} files)",
                                old.display()
                            ),
                            Err(e) => {
                                eprintln!("[prefix-cache] {}: {e}: not removed", old.display())
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&tmp);
                    eprintln!("[prefix-cache] {}: {e}: not written", path.display());
                }
            }
        })
    }
}

fn touch(path: &Path) {
    if let Ok(f) = std::fs::OpenOptions::new().append(true).open(path) {
        let _ = f.set_modified(SystemTime::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefix_reads_back_what_was_written_and_nothing_else() {
        let dir = std::env::temp_dir().join(format!("herbert-prefix-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let pc = PrefixCache::new(&dir, 2, "desc A".into()).unwrap();
        let (t1, kv1) = (
            vec![2u32, 105, 9731, 107],
            (0..1000u32).map(|i| i as u8).collect::<Vec<u8>>(),
        );
        assert!(pc.load(&t1).is_none());
        pc.store(t1.clone(), kv1.clone()).join().unwrap();
        assert_eq!(pc.load(&t1).as_deref(), Some(&kv1[..]));
        assert!(pc.load(&t1[..3]).is_none());
        // Another description (another build, device, tuning…): not read.
        let other = PrefixCache::new(&dir, 2, "desc B".into()).unwrap();
        assert!(other.load(&t1).is_none());
        assert!(other.newest().is_none());
        // A damaged file is ignored.
        let path = pc.path(&pc.key(&t1));
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(pc.load(&t1).is_none());
        // At most two files, the least recently used removed.
        for i in 0..3u32 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            pc.store(vec![2, i], vec![i as u8; 64]).join().unwrap();
        }
        assert_eq!(pc.listing().len(), 2);
        let (t, kv, _) = pc.newest().unwrap();
        assert_eq!((t, kv), (vec![2, 2], vec![2u8; 64]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The longest file that begins a request's tokens (all but the last), read back from the index
    /// a restart builds.
    #[test]
    fn the_longest_stored_prefix_of_a_request_is_found() {
        let dir = std::env::temp_dir().join(format!("herbert-prefix-best-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let pc = PrefixCache::new(&dir, 4, "desc".into()).unwrap();
        pc.store(vec![2, 10, 11], vec![1; 8]).join().unwrap();
        pc.store(vec![2, 10, 11, 12, 13], vec![2; 8])
            .join()
            .unwrap();
        pc.store(vec![2, 20], vec![3; 8]).join().unwrap();
        let ids = [2, 10, 11, 12, 13, 14];
        assert!(pc.has(&ids[..3]) && !pc.has(&ids[..4]));
        assert_eq!(pc.best(&ids, 0), Some(5));
        assert_eq!(pc.best(&ids, 5), None);
        assert_eq!(
            pc.best(&ids[..5], 0),
            Some(3),
            "the last token is never in the prefix"
        );
        let again = PrefixCache::new(&dir, 4, "desc".into()).unwrap();
        assert_eq!(again.best(&ids, 3), Some(5));
        assert_eq!(
            PrefixCache::new(&dir, 4, "other".into())
                .unwrap()
                .best(&ids, 0),
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
