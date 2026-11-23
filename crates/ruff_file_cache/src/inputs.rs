//! What a run read, recorded so that a later run can tell whether it would read the same:
//! by `stat` and, when that changed, by content.

use std::fs;
use std::io::{self, Read as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use crate::Shared;
use crate::digest::Digest;
use crate::stat::{Observation, Stat, Timestamp};

/// A path, and what was learned from it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Input {
    /// As the run named it: a UTF-8 path, absolute or relative to its current directory.
    path: PathBuf,
    /// What was learned from it.
    read: Read,
}

/// What a run learned from a path: what must read the same for its result to hold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Read {
    /// A directory listed: its `stat` changes when an entry is added, removed or renamed.
    Listing(Observation),
    /// A file read: its content decides what was read.
    Content {
        /// Its `stat` when it was read: while it holds, the file isn't read again.
        seen: Observation,
        /// The digest of what was read: what a file whose `stat` changed must still hash to.
        digest: Digest,
    },
    /// Looked for and not found.
    Missing,
    /// A directory whose existence, not listing, was read (a `.git` directory, whose own `stat`
    /// changes with every Git command).
    IsDir,
    /// A file whose existence, not content, was read (a path argument).
    IsFile,
}

impl Input {
    /// Returns what was learned from `path`.
    fn new(path: &Path, read: Read) -> Self {
        Self {
            path: path.to_path_buf(),
            read,
        }
    }

    /// Returns the file at `path`, read now. The `stat` is taken on the open file before reading
    /// it, so that a write during the read shows in the next run's `stat`.
    fn content(path: &Path, now: Timestamp) -> Option<Self> {
        let mut file = fs::File::open(path).ok()?;
        let meta = file.metadata().ok()?;
        if !meta.is_file() {
            return None;
        }
        let mut bytes = Vec::with_capacity(usize::try_from(meta.len()).unwrap_or(0));
        file.read_to_end(&mut bytes).ok()?;
        let seen = Observation::new(Stat::of(&meta), now);
        Some(Self::new(
            path,
            Read::Content {
                seen,
                digest: Digest::of_bytes(&bytes),
            },
        ))
    }

    /// Returns this input if it reads the same now, refreshed where only its `stat` changed.
    fn recheck(&self, now: Timestamp) -> Option<Self> {
        let path = &self.path;
        let holds = match &self.read {
            Read::Missing => {
                matches!(fs::symlink_metadata(path), Err(err) if err.kind() == io::ErrorKind::NotFound)
            }
            // Followed: recorded where a file was looked for, which follows symlinks too.
            Read::IsDir => fs::metadata(path).is_ok_and(|meta| meta.is_dir()),
            Read::IsFile => fs::symlink_metadata(path).is_ok_and(|meta| meta.is_file()),
            Read::Listing(seen) => Stat::of_path(path).is_some_and(|stat| seen.holds_for(&stat)),
            Read::Content { seen, digest } => {
                if Stat::of_target(path).is_some_and(|stat| seen.holds_for(&stat)) {
                    true
                } else {
                    return Self::content(path, now).filter(|now| {
                        matches!(&now.read, Read::Content { digest: new, .. } if new == digest)
                    });
                }
            }
        };
        holds.then(|| self.clone())
    }
}

/// What a run read, sorted by path.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inputs(Vec<Input>);

impl Inputs {
    /// Returns these inputs as they read now, checked in parallel, if they all read the same.
    pub fn recheck(&self, now: Timestamp) -> Option<Self> {
        self.0
            .par_iter()
            .map(|input| {
                let checked = input.recheck(now);
                if checked.is_none() {
                    tracing::debug!("Cache input changed: {}", input.path.display());
                }
                checked
            })
            .collect::<Option<Vec<_>>>()
            .map(Self)
    }

    /// Returns a digest of what these inputs read, not of when: the same for as long as they
    /// [recheck](Self::recheck) as holding. A result that depends on them includes it in its key.
    pub fn fingerprint(&self) -> Digest {
        /// What an input read, without when it was read.
        #[derive(Hash)]
        enum Seen<'a> {
            Listing(&'a Stat),
            Content(&'a Digest),
            Missing,
            IsDir,
            IsFile,
        }
        let seen: Vec<(&Path, Seen<'_>)> = self
            .0
            .iter()
            .map(|input| {
                let seen = match &input.read {
                    Read::Listing(seen) => Seen::Listing(seen.stat()),
                    Read::Content { digest, .. } => Seen::Content(digest),
                    Read::Missing => Seen::Missing,
                    Read::IsDir => Seen::IsDir,
                    Read::IsFile => Seen::IsFile,
                };
                (input.path.as_path(), seen)
            })
            .collect();
        Digest::of(&seen)
    }

    /// Returns the paths these inputs read, in order.
    pub fn paths(&self) -> impl Iterator<Item = &Path> {
        self.0.iter().map(|input| input.path.as_path())
    }
}

/// How a path was looked at: each is recorded once per path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Look {
    /// Its directory listing: [`Recorder::dir`].
    Listing,
    /// Its content, or that it's missing: [`Recorder::file`] and [`Recorder::file_if_present`].
    File,
    /// What kind of entry it is: [`Recorder::kind`].
    Kind,
}

/// Records what a run reads, from any of its threads.
#[derive(Debug)]
pub struct Recorder {
    /// When the run started: the time every `stat` it records counts as taken.
    now: Timestamp,
    /// What was recorded, in no particular order.
    inputs: Shared<Vec<Input>>,
    /// The paths recorded, by how: each is recorded once, however many threads look at it.
    seen: Shared<FxHashSet<(Look, PathBuf)>>,
    /// Set when something can't be recorded faithfully: the recording is then not kept.
    unusable: AtomicBool,
}

impl Recorder {
    /// Creates a recorder for a run that started at `now`, before it read anything.
    pub fn new(now: Timestamp) -> Self {
        Self {
            now,
            inputs: Shared::default(),
            seen: Shared::default(),
            unusable: AtomicBool::new(false),
        }
    }

    /// Marks the recording unusable: something about `path` can't be seen again from it.
    pub fn mark_unrecordable(&self, path: &Path) {
        tracing::debug!("Can't record cache input: {}", path.display());
        self.unusable.store(true, Ordering::Relaxed);
    }

    /// Returns whether `path` is looked at as `look` for the first time.
    fn first(&self, look: Look, path: &Path) -> bool {
        if path.to_str().is_none() {
            self.mark_unrecordable(path);
            return false;
        }
        self.seen.lock().insert((look, path.to_path_buf()))
    }

    /// Adds `input` to the recording.
    fn push(&self, input: Input) {
        self.inputs.lock().push(input);
    }

    /// Records a directory, `stat`-ed now: call before it's listed. A symbolic link can't be: its
    /// own `stat` doesn't change with its target's entries.
    pub fn dir(&self, path: &Path) {
        if !self.first(Look::Listing, path) {
            return;
        }
        let read = match fs::symlink_metadata(path) {
            Ok(meta) if meta.is_symlink() => return self.mark_unrecordable(path),
            Ok(meta) => Read::Listing(Observation::new(Stat::of(&meta), self.now)),
            Err(_) => Read::Missing,
        };
        self.push(Input::new(path, read));
    }

    /// Records a file read, or looked for: its content if it's there.
    pub fn file(&self, path: &Path) {
        if !self.first(Look::File, path) {
            return;
        }
        // Symlinks are followed, as reading the file follows them.
        match fs::metadata(path) {
            Ok(meta) if meta.is_file() => match Input::content(path, self.now) {
                Some(input) => self.push(input),
                None => self.mark_unrecordable(path),
            },
            Ok(meta) if meta.is_dir() => self.push(Input::new(path, Read::IsDir)),
            Ok(_) => self.mark_unrecordable(path),
            Err(_) => self.push(Input::new(path, Read::Missing)),
        }
    }

    /// Records what kind of entry `path` is: a file, a directory or missing, not its content.
    pub fn kind(&self, path: &Path) {
        if !self.first(Look::Kind, path) {
            return;
        }
        match fs::symlink_metadata(path) {
            Ok(meta) if meta.is_dir() => self.push(Input::new(path, Read::IsDir)),
            Ok(meta) if meta.is_file() => self.push(Input::new(path, Read::IsFile)),
            Ok(_) => self.mark_unrecordable(path),
            Err(_) => self.push(Input::new(path, Read::Missing)),
        }
    }

    /// Records a file only if it's there: for files whose appearance the enclosing directory's
    /// recorded listing shows.
    pub fn file_if_present(&self, path: &Path) {
        if fs::symlink_metadata(path).is_ok() {
            self.file(path);
        }
    }

    /// Returns the inputs, or `None` if something couldn't be recorded.
    pub fn finish(self) -> Option<Inputs> {
        if self.unusable.into_inner() {
            return None;
        }
        let mut inputs = self.inputs.into_inner();
        inputs.sort_by(|a, b| a.path.cmp(&b.path));
        Some(Inputs(inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A time well after anything the test writes, so nothing is racy.
    fn later() -> Timestamp {
        Timestamp::now() + std::time::Duration::from_secs(20)
    }

    fn record(f: impl FnOnce(&Recorder)) -> Inputs {
        let recorder = Recorder::new(later());
        f(&recorder);
        recorder.finish().unwrap()
    }

    #[test]
    fn a_fingerprint_is_the_same_while_inputs_hold_and_changes_with_what_was_read() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("plugin.js");
        let missing = tmp.path().join("missing.js");
        fs::write(&file, "a").unwrap();
        let inputs = record(|r| {
            r.file(&file);
            r.file(&missing);
            r.dir(tmp.path());
        });
        let first = inputs.fingerprint();

        // Rewritten with the same content: the `stat` changes, what was read doesn't.
        fs::write(&file, "a").unwrap();
        let rechecked = inputs.recheck(later()).unwrap();
        assert_ne!(rechecked, inputs, "refreshed where only the stat changed");
        assert_eq!(rechecked.fingerprint(), first);

        fs::write(&file, "b").unwrap();
        let changed = record(|r| {
            r.file(&file);
            r.file(&missing);
            r.dir(tmp.path());
        });
        assert_ne!(changed.fingerprint(), first, "another content");
        let fewer = record(|r| r.file(&file));
        assert_ne!(fewer.fingerprint(), changed.fingerprint(), "other files");
    }

    #[test]
    fn a_file_holds_until_its_content_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join(".gitignore");
        fs::write(&file, "a\n").unwrap();
        let inputs = record(|r| r.file(&file));
        assert!(inputs.recheck(later()).is_some());
        fs::write(&file, "a\n").unwrap();
        assert!(
            inputs.recheck(later()).is_some(),
            "the same content with a new stat"
        );
        fs::write(&file, "b\n").unwrap();
        assert!(inputs.recheck(later()).is_none());
    }

    #[test]
    fn a_directory_changes_when_an_entry_appears() {
        let tmp = tempfile::tempdir().unwrap();
        let inputs = record(|r| r.dir(tmp.path()));
        assert!(inputs.recheck(later()).is_some());
        // Directory timestamps may not move within the same tick: make sure they do.
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(tmp.path().join("new.ts"), "").unwrap();
        assert!(inputs.recheck(later()).is_none());
    }

    #[test]
    fn an_absent_file_changes_when_it_appears() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".prettierignore");
        let inputs = record(|r| r.file(&path));
        assert_eq!(inputs, Inputs(vec![Input::new(&path, Read::Missing)]));
        assert!(inputs.recheck(later()).is_some());
        fs::write(&path, "").unwrap();
        assert!(inputs.recheck(later()).is_none());
    }

    #[test]
    fn a_racy_directory_is_not_trusted() {
        let tmp = tempfile::tempdir().unwrap();
        let recorder = Recorder::new(Timestamp::now());
        recorder.dir(tmp.path());
        assert!(recorder.finish().unwrap().recheck(later()).is_none());
    }

    const DIRS: [&str; 2] = ["a", "a/b"];
    const NAMES: [&str; 3] = ["x.ts", "y.ts", ".gitignore"];
    const CONTENTS: [&str; 3] = ["one\n", "two\n", "three\n"];

    /// A change to the tree after it was recorded.
    #[derive(Debug, Clone)]
    enum Change {
        Write {
            dir: usize,
            name: usize,
            content: usize,
        },
        /// The same size, different bytes, the modification time put back: only `ctime` tells.
        SameSizeOldMtime {
            dir: usize,
            name: usize,
        },
        Remove {
            dir: usize,
            name: usize,
        },
        Rename {
            dir: usize,
            from: usize,
            to: usize,
        },
        /// An entry made and removed again: the listing reads the same, its `stat` doesn't.
        Touch {
            dir: usize,
        },
        Chmod {
            dir: usize,
            name: usize,
        },
    }

    use quickcheck::Arbitrary;

    impl Arbitrary for Change {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut pick = |n: usize| usize::arbitrary(g) % n;
            let (dir, name, other) = (pick(DIRS.len()), pick(NAMES.len()), pick(NAMES.len()));
            match pick(6) {
                0 => Self::Write {
                    dir,
                    name,
                    content: pick(CONTENTS.len()),
                },
                1 => Self::SameSizeOldMtime { dir, name },
                2 => Self::Remove { dir, name },
                3 => Self::Rename {
                    dir,
                    from: name,
                    to: other,
                },
                4 => Self::Touch { dir },
                _ => Self::Chmod { dir, name },
            }
        }
    }

    impl Change {
        fn apply(&self, root: &Path) {
            let at = |dir: usize, name: usize| root.join(DIRS[dir]).join(NAMES[name]);
            match *self {
                Self::Write { dir, name, content } => {
                    let _ = fs::write(at(dir, name), CONTENTS[content]);
                }
                Self::SameSizeOldMtime { dir, name } => {
                    let path = at(dir, name);
                    let (Ok(bytes), Ok(meta)) = (fs::read(&path), fs::metadata(&path)) else {
                        return;
                    };
                    // Fails on a file made read-only: then nothing changed.
                    let flipped: Vec<u8> = bytes.iter().map(|b| b ^ 1).collect();
                    if fs::write(&path, flipped).is_ok() {
                        let file = fs::File::options().write(true).open(&path).unwrap();
                        file.set_modified(meta.modified().unwrap()).unwrap();
                    }
                }
                Self::Remove { dir, name } => {
                    let _ = fs::remove_file(at(dir, name));
                }
                Self::Rename { dir, from, to } => {
                    let _ = fs::rename(at(dir, from), at(dir, to));
                }
                Self::Touch { dir } => {
                    let path = root.join(DIRS[dir]).join("touch.tmp");
                    fs::write(&path, "").unwrap();
                    fs::remove_file(&path).unwrap();
                }
                Self::Chmod { dir, name } => {
                    let path = at(dir, name);
                    if let Ok(meta) = fs::metadata(&path) {
                        let mut permissions = meta.permissions();
                        permissions.set_readonly(!permissions.readonly());
                        fs::set_permissions(&path, permissions).unwrap();
                    }
                }
            }
        }
    }

    /// What an input says was there, read again now: a listing's names, a file's bytes,
    /// or what kind of entry (if any) is at the path.
    fn reread(input: &Input) -> String {
        let path = &input.path;
        match &input.read {
            Read::Listing(_) => match fs::read_dir(path) {
                Ok(entries) => {
                    let mut names: Vec<_> = entries
                        .map(|e| e.unwrap().file_name().into_string().unwrap())
                        .collect();
                    names.sort();
                    format!("listing {names:?}")
                }
                Err(_) => "missing".into(),
            },
            Read::Content { .. } => match fs::read(path) {
                Ok(bytes) => format!("content {bytes:?}"),
                Err(_) => "missing".into(),
            },
            Read::Missing | Read::IsDir | Read::IsFile => match fs::symlink_metadata(path) {
                Ok(meta) if meta.is_dir() => "dir".into(),
                Ok(_) => "file".into(),
                Err(_) => "missing".into(),
            },
        }
    }

    /// For any tree and any changes to it: inputs that recheck as holding read what was read,
    /// and inputs of an unchanged tree hold.
    #[test]
    fn a_recheck_that_holds_reads_what_was_read() {
        #[expect(clippy::needless_pass_by_value, reason = "quickcheck passes values")]
        fn property(initial: Vec<(usize, usize, usize)>, changes: Vec<Change>) -> bool {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            for dir in DIRS {
                fs::create_dir_all(root.join(dir)).unwrap();
            }
            for (dir, name, content) in initial {
                let path = root
                    .join(DIRS[dir % DIRS.len()])
                    .join(NAMES[name % NAMES.len()]);
                fs::write(path, CONTENTS[content % CONTENTS.len()]).unwrap();
            }
            let inputs = record(|recorder| {
                for dir in DIRS {
                    recorder.dir(&root.join(dir));
                    for name in NAMES {
                        recorder.file(&root.join(dir).join(name));
                        recorder.kind(&root.join(dir).join(name));
                    }
                }
            });
            let read: Vec<String> = inputs.0.iter().map(reread).collect();
            if inputs.recheck(later()).is_none() {
                return false;
            }
            for change in &changes {
                change.apply(root);
            }
            match inputs.recheck(later()) {
                Some(_) => inputs.0.iter().map(reread).collect::<Vec<_>>() == read,
                None => true,
            }
        }
        quickcheck::QuickCheck::new()
            .tests(150)
            .quickcheck(property as fn(Vec<(usize, usize, usize)>, Vec<Change>) -> bool);
    }
}
