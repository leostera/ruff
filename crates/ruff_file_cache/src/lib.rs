//! Mechanisms for caching a command-line run's work across runs: remember what a run read and
//! found, so that a later run only redoes what changed.
//!
//! - [`Digest`]: BLAKE3 of file contents, and of the values keys are made of.
//! - [`Stat`] and [`Observation`]: a file's metadata and when it was taken, trusted only outside
//!   its racy window.
//! - [`Recorder`] and [`Inputs`]: what a run read (files, directory listings, paths it looked
//!   for), rechecked by a later run by `stat` and, when that changed, by content.
//! - A store of results by content in each [`CacheDir`], bounded by generations
//!   ([`Retention`]), shared by every cache in it.
//! - [`Cache`]: per-file results kept by `stat` or by content.
//! - [`CacheDir`]: where it's all kept, checksummed and written atomically.
//! - [`BuildId`]: the running binary, so one build's results are never used by another.
//!
//! What to cache, under which keys and where, is the caller's policy: these are the mechanisms.
//!
//! A cache never changes an answer by failing: anything missing, corrupt or unreadable is a miss,
//! and anything unwritable is skipped.
//!
//! ## Trusting a `stat`
//!
//! Each recorded file is compared with its own earlier `stat`, not with when the last run
//! happened (as Make does), so a tool that puts an old modification time back can't hide a
//! change. On Unix, the inode change time is part of the identity, and no tool can set it. A
//! `stat` taken within 2 s of the file's own timestamps is never trusted, because a write in the
//! same timestamp tick wouldn't show: the same rule as Git's racy index. A file whose `stat`
//! changed is hashed, and holds if its content didn't change.
//!
//! ## Recording inputs
//!
//! A [`Recorder`] is shared by a run's threads. It records a directory's `stat` before the
//! directory is listed, so that an entry added or removed later changes the directory. It records
//! a file's content as the file is read, and a path that was looked for and missing as missing.
//! If anything can't be recorded faithfully, [`Recorder::finish`] returns `None` and nothing
//! should be kept.
//!
//! ## Per-file results
//!
//! A [`Cache`] holds the last run's entries and collects this run's from any thread. An entry is
//! a file's path, the caller's key, the `stat` the file was read with, its content's digest and
//! the caller's result. A lookup only answers with an entry for the same key:
//! [`Cache::get_by_stat`] needs the same `stat`, outside the racy window, and
//! [`Cache::get_by_content`] needs the same content, or finds a result any cache in the
//! directory put in its store for the same key, path and content. [`Cache::save`] writes the
//! entries only if they changed, so a run that changed nothing writes nothing.

mod build;
mod cache;
mod digest;
mod dir;
mod inputs;
mod stat;
mod store;

pub use build::BuildId;
pub use cache::{Cache, Entry};
pub use digest::Digest;
pub use dir::CacheDir;
pub use inputs::{Inputs, Recorder};
pub use stat::{Observation, Stat, Timestamp};
pub use store::Retention;

use std::sync::{Mutex, MutexGuard, PoisonError};

/// State a run's threads share, behind a lock.
///
/// Everything kept this way is only ever added to, so a thread that panicked while holding the
/// lock leaves it consistent: the lock is recovered rather than poisoned.
#[derive(Debug, Default)]
struct Shared<T>(Mutex<T>);

impl<T> Shared<T> {
    /// Locks the value for one update.
    fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Returns the value, once no thread shares it.
    fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}
