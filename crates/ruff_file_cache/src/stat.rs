//! What a cache remembers of a file's or directory's metadata, and when to trust it.

use std::fs::Metadata;
use std::ops::{Add, Sub};
use std::path::Path;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// How long after a change a timestamp may still not tell it apart from the next change.
/// Coarser than every local file system's timestamps (FAT's are 2 s).
const RACY_WINDOW: Duration = Duration::from_secs(2);

/// A point in time: nanoseconds since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Returns the current time.
    pub fn now() -> Self {
        Self::of(SystemTime::now())
    }

    /// Returns `time` as nanoseconds since the Unix epoch (negative before it), saturating.
    fn of(time: SystemTime) -> Self {
        Self(match time.duration_since(SystemTime::UNIX_EPOCH) {
            Ok(after) => i64::try_from(after.as_nanos()).unwrap_or(i64::MAX),
            Err(before) => -i64::try_from(before.duration().as_nanos()).unwrap_or(i64::MAX),
        })
    }

    /// Returns the timestamp `secs` and `nanos` after the Unix epoch, saturating.
    #[cfg(unix)]
    fn from_parts(secs: i64, nanos: i64) -> Self {
        Self(secs.saturating_mul(1_000_000_000).saturating_add(nanos))
    }

    /// Returns `duration` in nanoseconds, saturating.
    fn nanos(duration: Duration) -> i64 {
        i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX)
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    fn add(self, duration: Duration) -> Self {
        Self(self.0.saturating_add(Self::nanos(duration)))
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Self;

    fn sub(self, duration: Duration) -> Self {
        Self(self.0.saturating_sub(Self::nanos(duration)))
    }
}

/// A file's identity and timestamps, as `stat` gives them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Stat {
    /// The size in bytes.
    size: u64,
    /// The modification time, which tools can set (and put back, as `rsync -t` and `cp -p` do).
    mtime: Timestamp,
    /// The inode change time on Unix, which no tool can set; the modification time elsewhere.
    ctime: Timestamp,
    /// The inode number on Unix, `0` elsewhere: a file replaced by a rename has another.
    ino: u64,
    /// The device on Unix, `0` elsewhere.
    dev: u64,
    /// The file type and permission bits on Unix; whether the file is read-only elsewhere.
    mode: u32,
}

impl Stat {
    /// Returns the `Stat` of `meta`.
    #[cfg(unix)]
    pub fn of(meta: &Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;
        Self {
            size: meta.size(),
            mtime: Timestamp::from_parts(meta.mtime(), meta.mtime_nsec()),
            ctime: Timestamp::from_parts(meta.ctime(), meta.ctime_nsec()),
            ino: meta.ino(),
            dev: meta.dev(),
            mode: meta.mode(),
        }
    }

    /// Returns the `Stat` of `meta`.
    #[cfg(not(unix))]
    pub fn of(meta: &Metadata) -> Self {
        let mtime = meta.modified().map_or(Timestamp(0), Timestamp::of);
        Self {
            size: meta.len(),
            mtime,
            ctime: mtime,
            ino: 0,
            dev: 0,
            mode: u32::from(meta.permissions().readonly()),
        }
    }

    /// Returns the `Stat` of the entry at `path`, not following a symlink.
    pub fn of_path(path: &Path) -> Option<Self> {
        std::fs::symlink_metadata(path)
            .ok()
            .map(|meta| Self::of(&meta))
    }

    /// Returns the `Stat` of the entry at `path`, following symlinks.
    pub fn of_target(path: &Path) -> Option<Self> {
        std::fs::metadata(path).ok().map(|meta| Self::of(&meta))
    }

    /// Returns the file type and permission bits on Unix; whether the file is read-only
    /// elsewhere.
    pub fn mode(&self) -> u32 {
        self.mode
    }

    /// Returns whether the entry changed at or after `time`.
    pub fn changed_since(&self, time: Timestamp) -> bool {
        self.mtime >= time || self.ctime >= time
    }
}

/// A [`Stat`], and when it was taken: the start of the run that took it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    /// The `stat` taken.
    stat: Stat,
    /// When it was taken. A `stat` is only trusted once its timestamps are older than this by
    /// the racy window.
    at: Timestamp,
}

impl Observation {
    /// Returns `stat`, as taken at `at`.
    pub fn new(stat: Stat, at: Timestamp) -> Self {
        Self { stat, at }
    }

    /// Returns the `stat` observed.
    pub fn stat(&self) -> &Stat {
        &self.stat
    }

    /// Returns when the `stat` was taken.
    pub fn at(&self) -> Timestamp {
        self.at
    }

    /// Returns whether an entry that now has `current` is the one observed: the same `stat`,
    /// observed late enough after its timestamps that a later change in the same timestamp tick
    /// would show. Racy, like Git's index: <https://git-scm.com/docs/racy-git>.
    pub fn holds_for(&self, current: &Stat) -> bool {
        self.stat == *current && !self.stat.changed_since(self.at - RACY_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(mtime: i64, ctime: i64) -> Stat {
        Stat {
            size: 1,
            mtime: Timestamp(mtime),
            ctime: Timestamp(ctime),
            ino: 1,
            dev: 1,
            mode: 0,
        }
    }

    #[test]
    fn holds_only_outside_the_racy_window() {
        let window = Timestamp::nanos(RACY_WINDOW);
        let at = Timestamp(10 * window);
        let old = at.0 - window - 1;
        let seen = |stat| Observation::new(stat, at);
        assert!(seen(stat(old, old)).holds_for(&stat(old, old)));
        assert!(
            !seen(stat(old, old)).holds_for(&stat(old, old + 1)),
            "a different stat"
        );
        assert!(!seen(stat(at.0 - window, old)).holds_for(&stat(at.0 - window, old)));
        assert!(
            !seen(stat(old, at.0 - 1)).holds_for(&stat(old, at.0 - 1)),
            "a recent ctime alone"
        );
        assert!(
            !seen(stat(at.0 + 5, old)).holds_for(&stat(at.0 + 5, old)),
            "after the observation"
        );
    }
}
