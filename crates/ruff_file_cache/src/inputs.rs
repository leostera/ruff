//! What a run read, recorded so that a later run can tell whether it would read the same:
//! by `stat` and, when that changed, by content.

use std::ffi::OsStr;
use std::fs;
use std::io::{self, Read as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
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
    /// A question asked of a path, and its answer.
    Answer {
        /// What was asked.
        question: Question,
        /// What the answer was.
        answer: bool,
    },
    /// Entries of a directory looked at, by name, and their kinds. They keep their kinds while
    /// the directory's `stat` holds, as an entry can't appear, go or change kind without changing
    /// it: many are checked by one `stat`.
    Entries {
        /// The directory's `stat`, taken before any of the entries was looked at.
        seen: Observation,
        /// The entries' names and kinds, by name.
        entries: Vec<(Box<str>, Kind)>,
    },
}

/// What kind of entry a directory has under a name, not following symbolic links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
enum Kind {
    /// No entry.
    Missing,
    /// A regular file.
    File,
    /// A directory.
    Dir,
}

impl Kind {
    /// Returns the kind of entry at `path`, or `None` for one its directory's `stat` can't vouch
    /// for: a symbolic link, whose target can change, a special file, or one that can't be read.
    fn of(path: &Path) -> Option<Self> {
        match fs::symlink_metadata(path) {
            Ok(meta) if meta.is_file() => Some(Self::File),
            Ok(meta) if meta.is_dir() => Some(Self::Dir),
            Ok(_) => None,
            Err(err) if err.kind() == io::ErrorKind::NotFound => Some(Self::Missing),
            Err(_) => None,
        }
    }

    /// Returns the answer to `question` for an entry of this kind.
    fn answers(self, question: Question) -> bool {
        match question {
            Question::IsDir => self == Self::Dir,
            Question::IsFile => self == Self::File,
        }
    }
}

/// A question a run asked of a path, as the standard library asks it: following symbolic links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Question {
    /// [`Path::is_dir`].
    IsDir,
    /// [`Path::is_file`].
    IsFile,
}

impl Question {
    /// Returns the answer for `path`, now.
    fn ask(self, path: &Path) -> bool {
        match self {
            Self::IsDir => path.is_dir(),
            Self::IsFile => path.is_file(),
        }
    }
}

impl Read {
    /// Returns whether this and `other` learned the same, if at different times.
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Listing(a), Self::Listing(b)) => a.stat() == b.stat(),
            (
                Self::Content { seen, digest },
                Self::Content {
                    seen: other_seen,
                    digest: other_digest,
                },
            ) => seen.stat() == other_seen.stat() && digest == other_digest,
            (
                Self::Entries { seen, entries },
                Self::Entries {
                    seen: other_seen,
                    entries: other_entries,
                },
            ) => seen.stat() == other_seen.stat() && entries == other_entries,
            _ => self == other,
        }
    }
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
            Read::Answer { question, answer } => question.ask(path) == *answer,
            Read::Entries { seen, entries } => {
                // The directory's `stat` first, so that an entry changed after it shows next time.
                let stat = Stat::of_path(path)?;
                if seen.holds_for(&stat) {
                    true
                } else {
                    let same = entries
                        .iter()
                        .all(|(name, kind)| Kind::of(&path.join(&**name)) == Some(*kind));
                    return same.then(|| {
                        let seen = Observation::new(stat, now);
                        let entries = entries.clone();
                        Self::new(path, Read::Entries { seen, entries })
                    });
                }
            }
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
        let recheck = |input: &Input| {
            let checked = input.recheck(now);
            if checked.is_none() {
                tracing::debug!("Cache input changed: {}", input.path.display());
            }
            checked
        };
        // Few inputs are checked faster than they're handed to other threads.
        if self.0.len() < Self::PARALLEL {
            self.0.iter().map(recheck).collect::<Option<_>>().map(Self)
        } else {
            self.0
                .par_iter()
                .map(recheck)
                .collect::<Option<_>>()
                .map(Self)
        }
    }

    /// How many inputs are worth rechecking in parallel.
    const PARALLEL: usize = 64;

    /// Returns whether no input was recorded.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
            Answer(Question, bool),
            Entries(&'a [(Box<str>, Kind)]),
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
                    Read::Answer { question, answer } => Seen::Answer(*question, *answer),
                    Read::Entries { entries, .. } => Seen::Entries(entries),
                };
                (input.path.as_path(), seen)
            })
            .collect();
        Digest::of(&seen)
    }

    /// Returns these inputs and `newer`, recorded since: what both learned the same once, as
    /// `newer` learned it.
    #[must_use]
    pub fn union(self, newer: Self) -> Self {
        if self.0.is_empty() {
            return newer;
        }
        let mut inputs = self.0;
        inputs.extend(newer.0);
        inputs.sort_by(|a, b| a.path.cmp(&b.path));
        let mut union: Vec<Input> = Vec::with_capacity(inputs.len());
        for input in inputs {
            let same_path = union
                .iter()
                .rposition(|kept| kept.path != input.path)
                .map_or(0, |last_other| last_other + 1);
            if let Some(kept) = union[same_path..]
                .iter_mut()
                .find(|kept| kept.read.same_as(&input.read))
            {
                *kept = input;
                continue;
            }
            // Entries of a directory seen with the same `stat` are one: of a directory unchanged
            // in between.
            if let Read::Entries { seen, entries } = &input.read && let Some(Read::Entries {
                seen: kept_seen,
                entries: kept,
            }) = union[same_path..]
                .iter_mut()
                .map(|kept| &mut kept.read)
                .find(
                    |kept| matches!(kept, Read::Entries { seen: s, .. } if s.stat() == seen.stat()),
                )
            {
                *kept_seen = *seen;
                kept.extend(entries.iter().cloned());
                // A name seen as two kinds stays twice: one won't read the same, as it should.
                kept.sort();
                kept.dedup();
                continue;
            }
            union.push(input);
        }
        Self(union)
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
    /// A question asked of it: [`Recorder::answer`].
    Answer(Question),
}

/// A directory whose entries a [`Recorder`] looked at.
#[derive(Debug)]
enum Dir {
    /// A directory, `stat`-ed before any of its entries was looked at, and those entries.
    Seen {
        /// Its `stat`.
        seen: Observation,
        /// The entries looked at, and their kinds.
        entries: FxHashMap<Box<str>, Kind>,
    },
    /// Missing: so are its entries, while it is.
    Missing,
    /// A symbolic link or a special file, whose `stat` can't vouch for entries.
    Unusable,
}

/// What a [`Dir`] is, without its entries.
#[derive(Debug, Clone, Copy)]
enum DirState {
    Seen,
    Missing,
    Unusable,
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
    /// The directories whose entries were recorded through their `stat`, and those entries.
    dirs: Shared<FxHashMap<PathBuf, Dir>>,
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
            dirs: Shared::default(),
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

    /// Records the `answer` to a `question` the run asked of `path`. The answer is the run's own,
    /// not asked again: whatever happens to `path` meanwhile, the run's result follows from it.
    ///
    /// Where the answer follows from what kind of entry `path` is (it's missing, or a file or a
    /// directory, not a symbolic link), that's recorded instead, through its directory's `stat`:
    /// the answers of most paths in a directory are then checked by one `stat`.
    pub fn answer(&self, path: &Path, question: Question, answer: bool) {
        if self.first(Look::Answer(question), path)
            && !self.entry(path, &|kind| kind.answers(question) == answer)
        {
            self.push(Input::new(path, Read::Answer { question, answer }));
        }
    }

    /// Records the kind of entry at `path` through its directory's `stat`, if it `fits`. Returns
    /// whether it did. An entry whose directory is missing is recorded as its directory is.
    fn entry(&self, path: &Path, fits: &dyn Fn(Kind) -> bool) -> bool {
        let (Some(dir), Some(name)) = (path.parent(), path.file_name().and_then(OsStr::to_str))
        else {
            return false;
        };
        // The directory's `stat` before the entry's: an entry can't then change unseen.
        match self.dir_state(dir) {
            DirState::Seen => {}
            DirState::Missing => {
                return fits(Kind::Missing) && self.entry(dir, &|kind| kind == Kind::Missing);
            }
            DirState::Unusable => return false,
        }
        let Some(kind) = Kind::of(path).filter(|kind| fits(*kind)) else {
            return false;
        };
        let mut dirs = self.dirs.lock();
        let Some(Dir::Seen { entries, .. }) = dirs.get_mut(dir) else {
            return false;
        };
        // Seen as another kind before, from another question: the answers are recorded as such.
        *entries.entry(name.into()).or_insert(kind) == kind
    }

    /// Returns what the directory at `path` is, `stat`-ing it the first time.
    fn dir_state(&self, path: &Path) -> DirState {
        let mut dirs = self.dirs.lock();
        let dir =
            dirs.entry(path.to_path_buf())
                .or_insert_with(|| match fs::symlink_metadata(path) {
                    Ok(meta) if meta.is_dir() => Dir::Seen {
                        seen: Observation::new(Stat::of(&meta), self.now),
                        entries: FxHashMap::default(),
                    },
                    Err(err) if err.kind() == io::ErrorKind::NotFound => Dir::Missing,
                    _ => Dir::Unusable,
                });
        match dir {
            Dir::Seen { .. } => DirState::Seen,
            Dir::Missing => DirState::Missing,
            Dir::Unusable => DirState::Unusable,
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
        inputs.extend(
            self.dirs
                .into_inner()
                .into_iter()
                .filter_map(|(path, dir)| {
                    let Dir::Seen { seen, entries } = dir else {
                        return None;
                    };
                    let mut entries: Vec<_> = entries.into_iter().collect();
                    entries.sort();
                    (!entries.is_empty())
                        .then(|| Input::new(&path, Read::Entries { seen, entries }))
                }),
        );
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
    fn an_answer_holds_until_the_question_is_answered_otherwise() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("pkg");
        let module = tmp.path().join("pkg.py");
        let inputs = record(|r| {
            r.answer(&dir, Question::IsDir, false);
            r.answer(&module, Question::IsFile, false);
        });
        assert!(inputs.recheck(later()).is_some());
        fs::write(&module, "").unwrap();
        assert!(inputs.recheck(later()).is_none());
        fs::remove_file(&module).unwrap();
        fs::create_dir(&dir).unwrap();
        assert!(inputs.recheck(later()).is_none());
        let answered = record(|r| r.answer(&dir, Question::IsDir, true));
        assert!(answered.recheck(later()).is_some());
        assert_ne!(answered.fingerprint(), inputs.fingerprint());
    }

    #[test]
    fn a_union_keeps_each_read_once() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("f");
        fs::write(&file, "x").unwrap();
        let (a, b) = (tmp.path().join("a"), tmp.path().join("b"));
        let older = record(|r| {
            r.file(&file);
            r.answer(&a, Question::IsDir, false);
            r.answer(&a, Question::IsFile, false);
        });
        let newer = record(|r| {
            r.file(&file);
            r.answer(&b, Question::IsDir, false);
        });
        let union = older.clone().union(newer.clone());
        // The file once, and the directory's entries `a` and `b`, seen with the same `stat`.
        assert_eq!(union.0.len(), 2, "{union:#?}");
        assert!(matches!(&union.0[0].read, Read::Entries { entries, .. } if entries.len() == 2));
        assert_eq!(union.clone().union(older).union(newer), union);
        assert_eq!(Inputs::default().union(union.clone()), union);
    }

    #[test]
    fn entries_are_checked_through_their_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let inputs = record(|r| {
            r.answer(&dir.join("pkg"), Question::IsDir, false);
            r.answer(&dir.join("pkg.py"), Question::IsFile, false);
            // Under a missing directory: recorded as the directory is.
            r.answer(&dir.join("src/pkg"), Question::IsDir, false);
        });
        assert_eq!(inputs.0.len(), 1, "one directory: {inputs:#?}");
        assert!(matches!(&inputs.0[0].read, Read::Entries { entries, .. } if entries.len() == 3));
        assert_eq!(inputs.recheck(later()), Some(inputs.clone()));

        // Another entry: the directory changed, its entries looked at hold, and it's refreshed.
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(dir.join("other.py"), "").unwrap();
        let refreshed = inputs.recheck(later()).unwrap();
        assert_ne!(refreshed, inputs);
        assert_eq!(refreshed.fingerprint(), inputs.fingerprint());

        fs::create_dir(dir.join("src")).unwrap();
        assert!(inputs.recheck(later()).is_none());
        fs::remove_dir(dir.join("src")).unwrap();
        fs::write(dir.join("pkg.py"), "").unwrap();
        assert!(inputs.recheck(later()).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_is_asked_again() {
        let tmp = tempfile::tempdir().unwrap();
        let (targets, links) = (tmp.path().join("targets"), tmp.path().join("links"));
        fs::create_dir_all(&targets).unwrap();
        fs::create_dir_all(&links).unwrap();
        let (to_dir, to_file) = (targets.join("dir"), targets.join("file"));
        fs::create_dir(&to_dir).unwrap();
        fs::write(&to_file, "").unwrap();
        std::os::unix::fs::symlink(&to_dir, links.join("dir")).unwrap();
        std::os::unix::fs::symlink(&to_file, links.join("file")).unwrap();
        let inputs = record(|r| {
            r.answer(&links.join("dir"), Question::IsDir, true);
            r.answer(&links.join("file"), Question::IsFile, true);
        });
        assert!(
            inputs
                .0
                .iter()
                .all(|input| matches!(input.read, Read::Answer { .. })),
            "{inputs:#?}"
        );
        // The targets swapped kinds: the links' directory didn't change, the answers did.
        fs::remove_file(&to_file).unwrap();
        fs::create_dir(&to_file).unwrap();
        assert!(inputs.recheck(later()).is_none());
        fs::remove_dir(&to_file).unwrap();
        fs::write(&to_file, "").unwrap();
        assert!(inputs.recheck(later()).is_some());
        fs::remove_dir(&to_dir).unwrap();
        fs::write(&to_dir, "").unwrap();
        assert!(inputs.recheck(later()).is_none());
    }

    #[test]
    fn a_path_that_changes_kind_during_a_run_isnt_vouched_for_by_its_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("module.py");
        let recorder = Recorder::new(later());
        recorder.answer(&path, Question::IsFile, false);
        fs::write(&path, "").unwrap();
        recorder.answer(&path, Question::IsDir, false);
        let inputs = recorder.finish().unwrap();
        // The run answered `is_file` with `false`: it must be asked again, and now differs.
        assert!(inputs.recheck(later()).is_none(), "{inputs:#?}");
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
        /// The entry replaced by a directory: another kind, under the same name.
        MakeDir {
            dir: usize,
            name: usize,
        },
    }

    use quickcheck::Arbitrary;

    impl Arbitrary for Change {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut pick = |n: usize| usize::arbitrary(g) % n;
            let (dir, name, other) = (pick(DIRS.len()), pick(NAMES.len()), pick(NAMES.len()));
            match pick(7) {
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
                5 => Self::MakeDir { dir, name },
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
                Self::MakeDir { dir, name } => {
                    let path = at(dir, name);
                    let _ = fs::remove_file(&path);
                    let _ = fs::create_dir(&path);
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
            Read::Answer { question, .. } => format!("{question:?} {}", question.ask(path)),
            Read::Entries { entries, .. } => {
                let kinds: Vec<_> = entries
                    .iter()
                    .map(|(name, _)| Kind::of(&path.join(&**name)))
                    .collect();
                format!("entries {kinds:?}")
            }
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
                        let path = root.join(dir).join(name);
                        recorder.file(&path);
                        recorder.kind(&path);
                        recorder.answer(&path, Question::IsDir, path.is_dir());
                        recorder.answer(&path, Question::IsFile, path.is_file());
                        let under = path.join("module");
                        recorder.answer(&under, Question::IsFile, under.is_file());
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
