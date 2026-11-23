//! A cache directory: its files are checksummed `postcard`, written atomically, and its store is
//! shared by every cache a run opens in it.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::Shared;
use crate::digest::Digest;
use crate::store::{Generation, Retention, Store};

/// Where the [`Store`] is kept in a [`CacheDir`].
const STORE: &str = "store";

/// A run's cache directory.
#[derive(Debug)]
pub struct CacheDir {
    /// Where it is: every name read or written is relative to it.
    path: PathBuf,
    /// The store, read the first time a cache looks a result up in it.
    store: OnceLock<Store>,
    /// What this run's caches used from the store and added to it: committed once, by
    /// [`Self::commit`].
    generation: Shared<Generation>,
}

impl CacheDir {
    /// Opens the directory at `path`, creating it if needed. Returns `None` if it can't be
    /// created.
    pub fn open(path: &Path) -> Option<Self> {
        fs::create_dir_all(path).ok()?;
        Some(Self {
            path: path.to_path_buf(),
            store: OnceLock::new(),
            generation: Shared::default(),
        })
    }

    /// Returns where the directory is.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads the value in `name`, a path relative to the directory.
    /// Returns `None` if it's missing or corrupt: a miss, never a wrong value.
    pub fn read<T: DeserializeOwned>(&self, name: &str) -> Option<T> {
        let bytes = fs::read(self.path.join(name)).ok()?;
        let (checksum, payload) = bytes.split_at_checked(32)?;
        if Digest::of_bytes(payload).as_ref() != checksum {
            tracing::debug!("Ignoring corrupt cache file `{name}`");
            return None;
        }
        postcard::from_bytes(payload).ok()
    }

    /// Writes `value` to `name` through a temporary file and a rename, so that readers see
    /// either the old or the new file. Failures are logged and otherwise ignored: a cache only
    /// saves work.
    pub fn write(&self, name: &str, value: &impl Serialize) {
        let Ok(payload) = postcard::to_stdvec(value) else {
            return;
        };
        let path = self.path.join(name);
        if let Err(err) = Self::write_atomically(&path, &payload) {
            tracing::debug!("Failed to write cache file `{}`: {err}", path.display());
        }
    }

    /// Writes the checksum of `payload` and `payload` to `path`, atomically.
    fn write_atomically(path: &Path, payload: &[u8]) -> io::Result<()> {
        let parent = path.parent().unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let mut file = Self::tempfile_in(parent)?;
        file.write_all(Digest::of_bytes(payload).as_ref())?;
        file.write_all(payload)?;
        file.persist(path)?;
        Ok(())
    }

    /// Returns a temporary file in `dir`, with the permissions a file created normally would
    /// have (`tempfile` defaults to `0o600`), so that a cache shared by several users stays
    /// writable by each.
    #[cfg(unix)]
    fn tempfile_in(dir: &Path) -> io::Result<tempfile::NamedTempFile> {
        use std::os::unix::fs::PermissionsExt;
        tempfile::Builder::new()
            .permissions(fs::Permissions::from_mode(0o666))
            .tempfile_in(dir)
    }

    /// Returns a temporary file in `dir`.
    #[cfg(not(unix))]
    fn tempfile_in(dir: &Path) -> io::Result<tempfile::NamedTempFile> {
        tempfile::Builder::new().tempfile_in(dir)
    }

    /// Removes all but the `keep` most recently written files in `subdir`.
    pub fn prune(&self, subdir: &str, keep: usize) {
        let Ok(entries) = fs::read_dir(self.path.join(subdir)) else {
            return;
        };
        let mut files: Vec<(SystemTime, PathBuf)> = entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                Some((entry.metadata().ok()?.modified().ok()?, entry.path()))
            })
            .collect();
        if files.len() > keep {
            files.sort_unstable_by_key(|(modified, _)| std::cmp::Reverse(*modified));
            for (_, path) in files.drain(keep..) {
                let _ = fs::remove_file(path);
            }
        }
    }

    /// Returns the value the store keeps for `key`, recording that this run used it.
    pub(crate) fn get_stored(&self, key: Digest) -> Option<&[u8]> {
        let value = self
            .store
            .get_or_init(|| self.read(STORE).unwrap_or_default())
            .get(&key)?;
        self.generation.lock().used.push(key);
        Some(value)
    }

    /// Adds `value` to the store under `key`, when this run commits.
    pub(crate) fn put_stored(&self, key: Digest, value: Box<[u8]>) {
        self.generation.lock().added.push((key, value));
    }

    /// Commits what this run's caches used from the store and added to it, as the store's next
    /// generation, bounded by `retention`. Called once, after every cache is saved.
    pub fn commit(&self, retention: Retention) {
        let generation = std::mem::take(&mut *self.generation.lock());
        if generation.is_empty() {
            return;
        }
        // Merged into the store as it is on disk now, so that runs in parallel lose as little as
        // possible.
        let mut store: Store = self.read(STORE).unwrap_or_default();
        if store.commit(&generation, retention) {
            self.write(STORE, &store);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_what_it_wrote_and_nothing_corrupt() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = CacheDir::open(tmp.path()).unwrap();
        dir.write("value", &(1u32, "a".to_string()));
        assert_eq!(
            dir.read::<(u32, String)>("value"),
            Some((1, "a".to_string()))
        );

        let path = dir.path.join("value");
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert_eq!(dir.read::<(u32, String)>("value"), None);
        fs::write(&path, &bytes[..10]).unwrap();
        assert_eq!(dir.read::<(u32, String)>("value"), None);
    }

    /// Whatever bytes of a file change, or however it's cut short, it reads as nothing:
    /// a corrupt cache is a miss, never a wrong value.
    #[test]
    fn any_corruption_reads_as_nothing() {
        #[expect(clippy::needless_pass_by_value, reason = "quickcheck passes values")]
        fn property(value: (Vec<String>, u64, bool), flip: (usize, u8), cut: usize) -> bool {
            let tmp = tempfile::tempdir().unwrap();
            let dir = CacheDir::open(tmp.path()).unwrap();
            dir.write("value", &value);
            if dir.read::<(Vec<String>, u64, bool)>("value").as_ref() != Some(&value) {
                return false;
            }
            let path = dir.path.join("value");
            let bytes = fs::read(&path).unwrap();
            let mut flipped = bytes.clone();
            let at = flip.0 % flipped.len();
            flipped[at] ^= flip.1.max(1);
            fs::write(&path, &flipped).unwrap();
            let corrupt = dir.read::<(Vec<String>, u64, bool)>("value").is_none();
            fs::write(&path, &bytes[..cut % bytes.len()]).unwrap();
            corrupt && dir.read::<(Vec<String>, u64, bool)>("value").is_none()
        }
        quickcheck::QuickCheck::new()
            .tests(300)
            .quickcheck(property as fn((Vec<String>, u64, bool), (usize, u8), usize) -> bool);
    }

    #[cfg(unix)]
    #[test]
    fn files_have_default_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let dir = CacheDir::open(tmp.path()).unwrap();
        dir.write("value", &1u8);
        let written = fs::metadata(tmp.path().join("value")).unwrap();
        fs::write(tmp.path().join("plain"), "").unwrap();
        let plain = fs::metadata(tmp.path().join("plain")).unwrap();
        assert_eq!(written.permissions().mode(), plain.permissions().mode());
    }

    #[test]
    fn prune_keeps_the_most_recently_written() {
        const KEEP: usize = 8;
        let tmp = tempfile::tempdir().unwrap();
        let dir = CacheDir::open(tmp.path()).unwrap();
        for i in 0..KEEP + 5 {
            dir.write(&format!("runs/{i}"), &i);
            let time = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(i as u64 + 1);
            fs::File::options()
                .write(true)
                .open(dir.path.join(format!("runs/{i}")))
                .unwrap()
                .set_modified(time)
                .unwrap();
        }
        dir.prune("runs", KEEP);
        assert_eq!(fs::read_dir(dir.path.join("runs")).unwrap().count(), KEEP);
        assert!(dir.read::<usize>("runs/4").is_none(), "the oldest go");
        assert_eq!(
            dir.read::<usize>(&format!("runs/{}", KEEP + 4)),
            Some(KEEP + 4)
        );
    }
}
