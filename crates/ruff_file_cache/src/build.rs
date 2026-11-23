//! Which build is running: its executable's digest, so that results kept by one build are never
//! used by another (every development build differs).
//!
//! Hashing the executable costs a read of it, so its digest is remembered by its path and `stat`.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::digest::Digest;
use crate::dir::CacheDir;
use crate::stat::Stat;

/// How many executables' digests are remembered, most recently hashed first. One cache directory
/// sees a few (an installed release, a pre-commit hook's, a development build), and alternating
/// between them shouldn't hash them again; but every new version adds one, so the oldest are
/// dropped. A forgotten executable is only hashed again.
const MAX_EXECUTABLES: usize = 16;

/// An executable whose digest is remembered.
#[derive(Serialize, Deserialize)]
struct Executable {
    /// Where it is, absolute.
    path: PathBuf,
    /// Its `stat` when it was hashed: an executable rebuilt in place has another.
    stat: Stat,
    /// The digest of its bytes.
    digest: Digest,
}

/// The running build: its executable's digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuildId(Digest);

impl BuildId {
    /// Returns the running build, or `None` if its executable can't be found or read: then
    /// nothing should be cached, as a result can't be tied to the code that produced it.
    /// Executables' digests are remembered in `file` of `dir`.
    pub fn current(dir: &CacheDir, file: &str) -> Option<Self> {
        let path = std::env::current_exe().ok()?;
        let stat = Stat::of_target(&path)?;
        let mut known: Vec<Executable> = dir.read(file).unwrap_or_default();
        if let Some(executable) = known
            .iter()
            .find(|executable| executable.path == path && executable.stat == stat)
        {
            return Some(Self(executable.digest));
        }
        let digest = Digest::of_bytes(&fs::read(&path).ok()?);
        known.retain(|executable| executable.path != path);
        known.insert(0, Executable { path, stat, digest });
        known.truncate(MAX_EXECUTABLES);
        dir.write(file, &known);
        Some(Self(digest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_and_remembers_this_executable() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = CacheDir::open(tmp.path()).unwrap();
        let first = BuildId::current(&dir, "builds").expect("the test binary is found");
        assert!(
            dir.read::<Vec<Executable>>("builds")
                .is_some_and(|known| known.len() == 1)
        );
        assert_eq!(
            BuildId::current(&dir, "builds"),
            Some(first),
            "the same executable, the same id"
        );
    }
}
