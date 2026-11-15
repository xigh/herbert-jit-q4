//! A model from Hugging Face given as `org/name` or `org/name@revision` (a branch, a tag or a
//! commit), found in the standard cache where `hf download` puts it: `$HF_HUB_CACHE`, else
//! `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, as `models--org--name/snapshots/<commit>/`
//! (`refs/<revision>` holds the commit). herbert does not download anything: a model missing from
//! the cache is an error that gives the `hf download` command to run.
//!
//! Cleaning the cache is left to the same tool, model by model (`hf cache ls`, `hf cache rm
//! model/org/name`, `hf cache prune` for the revisions no branch points to, each with `--dry-run`
//! first): the cache is shared with every other program that uses Hugging Face, so the whole folder
//! is not ours to delete. A service keeps the absolute path of its snapshot (`src/service.rs`):
//! uninstall it before removing its model.

use std::path::{Path, PathBuf};

/// `org/name[@revision]`: a repository of the Hub and the revision asked (default `main`).
#[derive(Debug, Clone, PartialEq)]
pub struct RepoId {
    pub repo: String,
    pub revision: String,
}

impl RepoId {
    /// `org/name` or `org/name@rev`: two names of letters, digits, `-`, `_` and `.`, one slash.
    pub fn parse(s: &str) -> Option<RepoId> {
        let (repo, revision) = match s.split_once('@') {
            Some((r, v)) if !v.is_empty() && !v.contains(char::is_whitespace) => (r, v),
            Some(_) => return None,
            None => (s, "main"),
        };
        let (org, name) = repo.split_once('/')?;
        let ok = |p: &str| {
            !p.is_empty()
                && !p.starts_with('.')
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        };
        (ok(org) && ok(name)).then(|| RepoId {
            repo: repo.to_string(),
            revision: revision.to_string(),
        })
    }
}

/// The model argument of the CLI and of the server: a directory that exists stays as it is (a
/// repository folder of the cache, `models--org--name`, gives its snapshot); else `org/name[@rev]`
/// is looked up in the cache and its snapshot returned; else the path as given (the loader says
/// what is wrong).
pub fn model_dir(arg: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(arg);
    if path.exists() {
        return crate::model::loader::resolve_model_dir(&path).map_err(|e| e.to_string());
    }
    match RepoId::parse(arg) {
        Some(id) => cached(&id),
        None => Ok(path),
    }
}

/// The files the engine reads, one pattern per `--include` (`hf` takes a second pattern after the
/// same `--include` for a file name, and then ignores the filter).
const INCLUDE: &str = "--include \"*.json\" --include \"*.safetensors\" --include \"*.jinja\" \
                       --include \"*.model\" --include \"merges.txt\" --include \"vocab.txt\"";

/// The snapshot of `id` in the cache, if it has a `config.json`.
pub fn cached(id: &RepoId) -> Result<PathBuf, String> {
    let repo_dir = cache_dir().join(format!("models--{}", id.repo.replace('/', "--")));
    let commit = std::fs::read_to_string(repo_dir.join("refs").join(&id.revision))
        .map(|c| c.trim().to_string())
        .unwrap_or_else(|_| id.revision.clone());
    let snap = repo_dir.join("snapshots").join(&commit);
    if snap.join("config.json").exists() {
        return Ok(snap);
    }
    let revision = if id.revision == "main" {
        String::new()
    } else {
        format!(" --revision {}", id.revision)
    };
    Err(format!(
        "{}@{} is not in the Hugging Face cache ({}): download it first with\n  hf download {}{revision} {INCLUDE}\n(the README says how to install `hf`)",
        id.repo,
        id.revision,
        cache_dir().display(),
        id.repo
    ))
}

/// The name the server reports for a model given as `org/name[@rev]` (`--alias` unless one is
/// given): the repository's, instead of the snapshot directory's (a commit). None for a directory
/// that exists.
pub fn alias_for(arg: &str) -> Option<String> {
    if Path::new(arg).exists() {
        return None;
    }
    RepoId::parse(arg).map(|id| id.repo)
}

/// The cache of the Hub: `$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`.
pub fn cache_dir() -> PathBuf {
    if let Some(d) = env("HF_HUB_CACHE") {
        return PathBuf::from(d);
    }
    env("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".cache").join("huggingface"))
        .join("hub")
}

fn home() -> PathBuf {
    env("HOME")
        .or_else(|| env("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        assert_eq!(
            RepoId::parse("Qwen/Qwen3.5-0.8B"),
            Some(RepoId {
                repo: "Qwen/Qwen3.5-0.8B".into(),
                revision: "main".into()
            })
        );
        assert_eq!(
            RepoId::parse("google/gemma-4-E2B-it@abc123")
                .unwrap()
                .revision,
            "abc123"
        );
        for bad in [
            "model",
            "./a/b",
            "/abs/path",
            "a/b/c",
            "a/",
            "/b",
            "a/b@",
            "C:\\m\\x",
            "../m",
        ] {
            assert_eq!(RepoId::parse(bad), None, "{bad}");
        }
    }
}
