//! A cache of per-file results: a file whose `stat` is unchanged keeps its last result without
//! being read; one that changed is read and keeps a result put for the same content, by the last
//! run or by any run of any cache in the same [`CacheDir`].
//!
//! A result is kept under a key the caller makes of everything it depends on besides the file's
//! path and content: the tool's build and settings, and the
//! [fingerprint](crate::Inputs::fingerprint) of anything else it read. A result that can't be
//! keyed so mustn't be put.
//!
//! What results depend on but is only known once they're computed, and is the same for most of
//! them, is recorded once for the whole cache instead: its [shared inputs](Cache::shared). The
//! results hold while those read the same, rechecked once per run; when one doesn't, every
//! result of the cache is dropped. The cache's [generation](Cache::generation), which the caller
//! puts in its keys, changes then, so that results put in the store before aren't found either.

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::Shared;
use crate::digest::Digest;
use crate::dir::CacheDir;
use crate::inputs::{Inputs, Recorder};
use crate::stat::{Observation, Stat, Timestamp};

/// A file's result, as a run put it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry<R> {
    /// The path, as the caller names it.
    pub path: Box<str>,
    /// Everything the result depends on besides the file's path and content.
    pub key: Digest,
    /// The file's `stat` when it was read.
    pub seen: Observation,
    /// The digest of the content the result is for.
    pub content: Digest,
    /// What the caller computed for that content, under that key.
    pub result: R,
}

/// A cache as a run saves it.
#[derive(Serialize, Deserialize)]
struct Saved<E> {
    /// The generation the entries were put in.
    generation: Digest,
    /// What every entry depends on besides what its key covers.
    shared: Inputs,
    /// The entries, by path.
    entries: Vec<E>,
}

/// One of the last run's entries.
struct Last<R> {
    /// The entry, as the last run saved it.
    entry: Entry<R>,
    /// Whether this run kept it by its `stat`: set without a lock or a copy, as every file of a
    /// run that changed nothing is kept so.
    kept: AtomicBool,
}

/// Per-file results kept across runs, shared by a run's threads.
pub struct Cache<R> {
    /// The directory this cache, its siblings and their shared store are in.
    dir: Arc<CacheDir>,
    /// The file in `dir` this cache's entries are kept in.
    name: String,
    /// When the run started: the time every `stat` it records counts as taken.
    started: Timestamp,
    /// How long an entry this run didn't look up is kept after it was last put.
    keep_unvisited: Duration,
    /// The last run's entries, by path.
    last: FxHashMap<Box<str>, Last<R>>,
    /// This run's entries but those [kept](Last::kept) by `stat`: with the kept ones, what the
    /// next run reads as its `last`.
    found: Shared<Vec<Entry<R>>>,
    /// Identifies the shared inputs the entries depend on: it changes when one reads otherwise.
    generation: Digest,
    /// The last run's shared inputs, which read the same now, as rechecked.
    last_shared: Inputs,
    /// The shared inputs this run's results depend on.
    shared: Recorder,
}

impl<R> fmt::Debug for Cache<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cache")
            .field("name", &self.name)
            .field("last", &self.last.len())
            .finish_non_exhaustive()
    }
}

impl<R: Clone + Serialize + DeserializeOwned> Cache<R> {
    /// Opens the cache kept in `name` of `dir`, with the last run's results, for a run that
    /// started at `started`: before it read any file.
    pub fn open(dir: Arc<CacheDir>, name: impl Into<String>, started: Timestamp) -> Self {
        let name = name.into();
        let saved: Option<Saved<Entry<R>>> = dir.read(&name);
        let previous = saved.as_ref().map(|saved| saved.generation);
        let (generation, last_shared, last) = match saved.and_then(|saved| {
            let shared = saved.shared.recheck(started)?;
            Some((saved.generation, shared, saved.entries))
        }) {
            Some(last) => last,
            // A new generation, unlike any before it.
            None => (
                Digest::of(&(&name, started, previous)),
                Inputs::default(),
                Vec::new(),
            ),
        };
        Self {
            last: last
                .into_iter()
                .map(|entry| {
                    let last = Last {
                        entry,
                        kept: AtomicBool::new(false),
                    };
                    (last.entry.path.clone(), last)
                })
                .collect(),
            dir,
            name,
            started,
            keep_unvisited: Duration::ZERO,
            found: Shared::default(),
            generation,
            last_shared,
            shared: Recorder::new(started),
        }
    }

    /// Returns the generation of the shared inputs: the caller puts it in its keys.
    pub fn generation(&self) -> Digest {
        self.generation
    }

    /// Returns where to record the shared inputs of this run's results.
    pub fn shared(&self) -> &Recorder {
        &self.shared
    }

    /// Keeps the entries of files this run didn't look up for `age` after they were last put,
    /// for caches whose runs each look at some of their files (the files given on the command
    /// line). By default, a run's entries are only the files it looked up.
    #[must_use]
    pub fn keep_unvisited_for(mut self, age: Duration) -> Self {
        self.keep_unvisited = age;
        self
    }

    /// Returns the last run's entry for `path`.
    pub fn entry(&self, path: &Path) -> Option<&Entry<R>> {
        self.last.get(path.to_str()?).map(|last| &last.entry)
    }

    /// Returns the last run's result for the file at `path`, which now has `stat`, under `key`,
    /// if the file is unchanged by its `stat`. Such a file needn't be read. A path that isn't
    /// UTF-8 is never cached.
    pub fn get_by_stat(&self, path: &Path, key: Digest, stat: Stat) -> Option<R> {
        let last = self.last.get(path.to_str()?)?;
        if last.entry.key != key || !last.entry.seen.holds_for(&stat) {
            return None;
        }
        last.kept.store(true, Ordering::Relaxed);
        Some(last.entry.result.clone())
    }

    /// Returns the result for the file at `path`, read with `stat` and hashing to `content`,
    /// under `key`: the last run's, or one any run of any cache in the directory put for that
    /// key, path and content.
    pub fn get_by_content(
        &self,
        path: &Path,
        key: Digest,
        stat: Stat,
        content: Digest,
    ) -> Option<R> {
        let name = path.to_str()?;
        let result = match self.last.get(name).map(|last| &last.entry) {
            Some(last) if last.key == key && last.content == content => last.result.clone(),
            _ => postcard::from_bytes(self.dir.get_stored(Self::store_key(name, key, content))?)
                .ok()?,
        };
        self.push(name, key, stat, content, result.clone());
        Some(result)
    }

    /// Puts `result` for the file at `path`, read with `stat` and hashing to `content`, under
    /// `key`, for this cache's next run and any cache's run.
    pub fn put(&self, path: &Path, key: Digest, stat: Stat, content: Digest, result: R) {
        let Some(name) = path.to_str() else {
            return;
        };
        if let Ok(value) = postcard::to_allocvec(&result) {
            self.dir
                .put_stored(Self::store_key(name, key, content), value.into());
        }
        self.push(name, key, stat, content, result);
    }

    /// Saves this run's results, unless they're the last run's. Results added to the store are
    /// written when the directory [commits](CacheDir::commit) it.
    pub fn save(self) {
        // Results that depend on what couldn't be recorded can't be kept, nor can those in the
        // store under this generation: the next run starts a new one.
        let Some(shared) = self.shared.finish() else {
            let saved = Saved::<Entry<R>> {
                generation: Digest::of(&(&self.name, self.started, self.generation)),
                shared: Inputs::default(),
                entries: Vec::new(),
            };
            self.dir.write(&self.name, &saved);
            return;
        };
        let shared = self.last_shared.clone().union(shared);
        let found = self.found.into_inner();
        let found_paths: FxHashSet<&str> = found.iter().map(|entry| &*entry.path).collect();
        let unvisited_since = self.started - self.keep_unvisited;
        // The last run's entries this run kept by `stat`, or didn't look up and still keeps.
        // A file found again is as found.
        let retained: Vec<&Entry<R>> = self
            .last
            .values()
            .filter(|last| {
                last.kept.load(Ordering::Relaxed)
                    || (!found_paths.contains(&*last.entry.path)
                        && last.entry.seen.at() >= unvisited_since)
            })
            .map(|last| &last.entry)
            .filter(|entry| !found_paths.contains(&*entry.path))
            .collect();
        // Nothing found, nothing dropped and nothing else depended on: the last run's entries,
        // as they are.
        if found.is_empty() && retained.len() == self.last.len() && shared == self.last_shared {
            return;
        }
        let mut entries: Vec<&Entry<R>> = retained.into_iter().chain(&found).collect();
        entries.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        entries.dedup_by(|a, b| a.path == b.path);
        let saved = Saved {
            generation: self.generation,
            shared,
            entries,
        };
        self.dir.write(&self.name, &saved);
    }

    /// Returns a result's key in the store, shared by every cache in the directory: the caller's
    /// key, the path and the content, so that a hit is for the same file, contents and
    /// dependencies.
    fn store_key(path: &str, key: Digest, content: Digest) -> Digest {
        Digest::of(&(key, path, content))
    }

    /// Adds this run's entry for a file read with `stat`, observed at the start of the run.
    fn push(&self, path: &str, key: Digest, stat: Stat, content: Digest, result: R) {
        let seen = Observation::new(stat, self.started);
        self.found.lock().push(Entry {
            path: path.into(),
            key,
            seen,
            content,
            result,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};

    use super::*;
    use crate::store::Retention;

    const CONTENTS: [&str; 3] = ["aaaa", "bbbb", "cc"];

    /// What a tool computes for `content` under `key`: what every answer must equal. It differs
    /// by key, so that an answer for another key is caught.
    fn compute(key: u8, content: &[u8]) -> bool {
        (usize::from(key) + usize::from(content[0])) % 2 == 0
    }

    #[derive(Debug, Clone, Copy)]
    enum Step {
        /// Write one of [`CONTENTS`].
        Write(usize),
        /// Write one of [`CONTENTS`] and put the modification time back: only `ctime` tells.
        WriteKeepingMtime(usize),
        /// Run under one of two keys.
        Run(u8),
        /// Forget the last run, as another cache in the directory would: only the store answers.
        Forget,
    }

    impl Arbitrary for Step {
        fn arbitrary(g: &mut Gen) -> Self {
            let content = usize::arbitrary(g) % CONTENTS.len();
            match u8::arbitrary(g) % 6 {
                0 => Self::Write(content),
                1 => Self::WriteKeepingMtime(content),
                2 => Self::Forget,
                _ => Self::Run(u8::arbitrary(g) % 2),
            }
        }
    }

    /// One run: every answer, from the `stat`, the last run or the store, is what computing
    /// gives.
    fn run(root: &Path, path: &Path, key: u8) -> Option<&'static str> {
        // Past the racy window, so that a `stat` can be trusted.
        let started = Timestamp::now() + Duration::from_secs(10);
        let dir = Arc::new(CacheDir::open(root).unwrap());
        let cache = Cache::open(Arc::clone(&dir), "files", started);
        let source = fs::read(path).unwrap();
        let expected = compute(key, &source);
        let (key, stat) = (Digest::of(&key), Stat::of_path(path).unwrap());
        let wrong = if let Some(result) = cache.get_by_stat(path, key, stat) {
            (result != expected).then_some("stale result by stat")
        } else {
            let content = Digest::of_bytes(&source);
            if let Some(result) = cache.get_by_content(path, key, stat, content) {
                (result != expected).then_some("wrong result by content")
            } else {
                cache.put(path, key, stat, content, expected);
                None
            }
        };
        cache.save();
        dir.commit(Retention::default());
        wrong
    }

    #[test]
    fn every_answer_is_what_computing_gives() {
        fn property(steps: Vec<Step>) -> TestResult {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().join("cache");
            let path = tmp.path().join("f");
            fs::write(&path, CONTENTS[0]).unwrap();
            for step in steps {
                match step {
                    Step::Write(i) => fs::write(&path, CONTENTS[i]).unwrap(),
                    Step::WriteKeepingMtime(i) => {
                        let mtime = fs::metadata(&path).unwrap().modified().unwrap();
                        fs::write(&path, CONTENTS[i]).unwrap();
                        let file = fs::File::options().write(true).open(&path).unwrap();
                        file.set_modified(mtime).unwrap();
                    }
                    Step::Forget => {
                        let _ = fs::remove_file(root.join("files"));
                    }
                    Step::Run(key) => {
                        if let Some(wrong) = run(&root, &path, key) {
                            return TestResult::error(wrong);
                        }
                    }
                }
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(200)
            .rng(Gen::new(20))
            .quickcheck(property as fn(Vec<Step>) -> TestResult);
    }

    #[test]
    fn another_cache_in_the_directory_finds_a_result_by_key_and_content() {
        let tmp = tempfile::tempdir().unwrap();
        let (f, g) = (tmp.path().join("f"), tmp.path().join("g"));
        fs::write(&f, "x").unwrap();
        fs::write(&g, "x").unwrap();
        let (stat, content) = (Stat::of_path(&f).unwrap(), Digest::of_bytes(b"x"));
        let now = Timestamp::now();

        let dir = Arc::new(CacheDir::open(tmp.path()).unwrap());
        let cache = Cache::open(Arc::clone(&dir), "a", now);
        cache.put(&f, Digest::of(&1), stat, content, 7u8);
        cache.save();
        dir.commit(Retention::default());

        let dir = Arc::new(CacheDir::open(tmp.path()).unwrap());
        let cache = Cache::<u8>::open(Arc::clone(&dir), "b", now);
        assert_eq!(
            cache.get_by_content(&f, Digest::of(&1), stat, content),
            Some(7)
        );
        assert_eq!(
            cache.get_by_content(&g, Digest::of(&1), stat, content),
            None,
            "another path"
        );
        assert_eq!(
            cache.get_by_content(&f, Digest::of(&2), stat, content),
            None,
            "another key"
        );
        let other = Digest::of_bytes(b"y");
        assert_eq!(
            cache.get_by_content(&f, Digest::of(&1), stat, other),
            None,
            "another content"
        );
    }

    #[test]
    fn a_run_that_changed_nothing_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("f");
        fs::write(&f, "x").unwrap();
        let stat = Stat::of_path(&f).unwrap();
        let later = Timestamp::now() + Duration::from_secs(10);
        let dir = Arc::new(CacheDir::open(&tmp.path().join("cache")).unwrap());
        let cache = Cache::open(Arc::clone(&dir), "files", later);
        cache.put(&f, Digest::of(&1), stat, Digest::of_bytes(b"x"), 1u8);
        cache.save();

        // Dated back, so that a rewrite would show.
        let written = dir.path().join("files");
        let old = std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let file = fs::File::options().write(true).open(&written).unwrap();
        file.set_modified(old).unwrap();

        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later);
        assert_eq!(cache.get_by_stat(&f, Digest::of(&1), stat), Some(1));
        cache.save();
        assert_eq!(fs::metadata(&written).unwrap().modified().unwrap(), old);
    }

    #[test]
    fn results_hold_while_their_shared_inputs_read_the_same() {
        use crate::Question;

        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("f");
        let module = tmp.path().join("module.py");
        fs::write(&f, "x").unwrap();
        let stat = Stat::of_path(&f).unwrap();
        let content = Digest::of_bytes(b"x");
        let later = Timestamp::now() + Duration::from_secs(10);
        let dir = Arc::new(CacheDir::open(&tmp.path().join("cache")).unwrap());
        let key = |cache: &Cache<u8>| Digest::of(&(1, cache.generation()));

        let cache = Cache::open(Arc::clone(&dir), "files", later);
        cache.put(&f, key(&cache), stat, content, 1u8);
        cache.shared().answer(&module, Question::IsFile, false);
        cache.save();
        dir.commit(Retention::default());

        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later);
        assert_eq!(cache.get_by_stat(&f, key(&cache), stat), Some(1));
        cache.save();

        // An answer changes: every result goes, the store's too, and a new generation starts.
        fs::write(&module, "").unwrap();
        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later);
        assert_eq!(cache.get_by_stat(&f, key(&cache), stat), None);
        assert_eq!(cache.get_by_content(&f, key(&cache), stat, content), None);
        cache.put(&f, key(&cache), stat, content, 2u8);
        cache.save();
        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later);
        assert_eq!(cache.get_by_stat(&f, key(&cache), stat), Some(2));
    }

    #[test]
    fn results_whose_shared_inputs_cant_be_recorded_arent_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("f");
        fs::write(&f, "x").unwrap();
        let stat = Stat::of_path(&f).unwrap();
        let content = Digest::of_bytes(b"x");
        let later = Timestamp::now() + Duration::from_secs(10);
        let dir = Arc::new(CacheDir::open(&tmp.path().join("cache")).unwrap());
        let key = |cache: &Cache<u8>| Digest::of(&(1, cache.generation()));

        let cache = Cache::open(Arc::clone(&dir), "files", later);
        cache.put(&f, key(&cache), stat, content, 1u8);
        cache.shared().mark_unrecordable(&f);
        cache.save();
        dir.commit(Retention::default());

        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later);
        assert_eq!(cache.get_by_stat(&f, key(&cache), stat), None);
        assert_eq!(cache.get_by_content(&f, key(&cache), stat, content), None);
    }

    #[test]
    fn unvisited_entries_are_kept_only_when_asked() {
        let tmp = tempfile::tempdir().unwrap();
        let (f, g) = (tmp.path().join("f"), tmp.path().join("g"));
        fs::write(&f, "x").unwrap();
        fs::write(&g, "x").unwrap();
        let (stat, content) = (Stat::of_path(&f).unwrap(), Digest::of_bytes(b"x"));
        let dir = Arc::new(CacheDir::open(&tmp.path().join("cache")).unwrap());
        let now = Timestamp::now();
        let cache = Cache::open(Arc::clone(&dir), "files", now);
        cache.put(&f, Digest::of(&1), stat, content, 1u8);
        cache.put(&g, Digest::of(&1), stat, content, 1u8);
        cache.save();

        let later = now + Duration::from_secs(60);
        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later)
            .keep_unvisited_for(Duration::from_secs(120));
        cache.put(&f, Digest::of(&2), stat, content, 2u8);
        cache.save();
        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later);
        assert_eq!(cache.entry(&g).map(|entry| entry.result), Some(1), "kept");
        assert_eq!(
            cache.entry(&f).map(|entry| entry.result),
            Some(2),
            "replaced"
        );

        cache.put(&f, Digest::of(&3), stat, content, 3u8);
        cache.save();
        let cache = Cache::<u8>::open(Arc::clone(&dir), "files", later);
        assert_eq!(cache.entry(&g), None, "dropped by default");
    }

    #[cfg(unix)]
    #[test]
    fn a_path_that_isnt_utf8_is_never_cached() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let tmp = tempfile::tempdir().unwrap();
        let dir = Arc::new(CacheDir::open(tmp.path()).unwrap());
        let file = tmp.path().join("f");
        fs::write(&file, "x").unwrap();
        let (stat, content) = (Stat::of_path(&file).unwrap(), Digest::of_bytes(b"x"));
        let path = Path::new(OsStr::from_bytes(b"\xff"));
        let cache = Cache::open(Arc::clone(&dir), "files", Timestamp::now());
        cache.put(path, Digest::of(&1), stat, content, 1u8);
        assert_eq!(
            cache.get_by_content(path, Digest::of(&1), stat, content),
            None
        );
        cache.save();
        assert!(
            Cache::<u8>::open(dir, "files", Timestamp::now())
                .last
                .is_empty()
        );
    }
}
