//! The questions linting a file asks of the file system, besides reading the file: a result that
//! is kept across runs holds only while each gets the same answer.
//!
//! Today that's `isort`'s, deciding whether a module is first-party: whether `<src>/<module>` is a
//! directory, or `<src>/<module>.py` or `.pyi` a file.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

/// A question asked of a path, following symbolic links.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Question {
    /// [`Path::is_dir`].
    IsDir,
    /// [`Path::is_file`].
    IsFile,
}

/// A question linting asked of a path, and its answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// The path asked about.
    pub path: PathBuf,
    /// What was asked.
    pub question: Question,
    /// The answer.
    pub answer: bool,
}

thread_local! {
    /// The probes of the [`Probes`] recording on this thread, if any.
    static PROBES: RefCell<Option<Vec<Probe>>> = const { RefCell::new(None) };
}

/// A recording of the questions linting asks of the file system on this thread, from
/// [`Probes::start`] until [`Probes::finish`]. Linting a file runs on one thread, so this records
/// all of them.
#[must_use]
pub struct Probes {
    /// The recording this one interrupted, resumed with this one's probes when it finishes.
    outer: Option<Vec<Probe>>,
    /// Whether it finished: the interrupted recording is resumed once.
    finished: bool,
}

impl Probes {
    /// Starts recording.
    pub fn start() -> Self {
        Self {
            outer: PROBES.replace(Some(Vec::new())),
            finished: false,
        }
    }

    /// Stops recording, and returns the questions asked since [`Probes::start`], in the order
    /// asked.
    pub fn finish(mut self) -> Vec<Probe> {
        self.stop()
    }

    /// Resumes the interrupted recording, adding this one's probes to it, and returns them.
    fn stop(&mut self) -> Vec<Probe> {
        self.finished = true;
        let probes = PROBES.replace(self.outer.take()).unwrap_or_default();
        PROBES.with_borrow_mut(|outer| {
            if let Some(outer) = outer {
                outer.extend(probes.iter().cloned());
            }
        });
        probes
    }

    /// Returns [`Path::is_dir`], recorded.
    pub(crate) fn is_dir(path: &Path) -> bool {
        Self::ask(path, Question::IsDir, Path::is_dir)
    }

    /// Returns [`Path::is_file`], recorded.
    pub(crate) fn is_file(path: &Path) -> bool {
        Self::ask(path, Question::IsFile, Path::is_file)
    }

    /// Returns `ask(path)`, recorded as the answer to `question` if a recording is running.
    fn ask(path: &Path, question: Question, ask: fn(&Path) -> bool) -> bool {
        let answer = ask(path);
        PROBES.with_borrow_mut(|probes| {
            if let Some(probes) = probes {
                probes.push(Probe {
                    path: path.to_path_buf(),
                    question,
                    answer,
                });
            }
        });
        answer
    }
}

impl Drop for Probes {
    fn drop(&mut self) {
        // Unfinished, as when linting returns early: still resume the interrupted recording.
        if !self.finished {
            self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Probe, Probes, Question};

    #[test]
    fn records_what_was_asked_and_answered() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let missing = dir.join("missing.py");
        let probes = Probes::start();
        assert!(Probes::is_dir(&dir));
        assert!(!Probes::is_file(&missing));
        assert_eq!(
            probes.finish(),
            [
                Probe {
                    path: dir,
                    question: Question::IsDir,
                    answer: true,
                },
                Probe {
                    path: missing,
                    question: Question::IsFile,
                    answer: false,
                },
            ]
        );
    }

    #[test]
    fn an_outer_recording_sees_what_an_inner_one_saw() {
        let tmp = tempfile::tempdir().unwrap();
        let outer = Probes::start();
        let inner = Probes::start();
        Probes::is_dir(tmp.path());
        let inner = inner.finish();
        let dropped = Probes::start();
        Probes::is_file(tmp.path());
        drop(dropped);
        let outer = outer.finish();
        assert_eq!(inner.len(), 1);
        assert_eq!(outer.len(), 2);
        assert_eq!(outer[..1], inner);
    }

    #[test]
    fn nothing_is_recorded_without_a_recording() {
        let tmp = tempfile::tempdir().unwrap();
        Probes::is_dir(tmp.path());
        assert!(Probes::start().finish().is_empty());
    }
}
