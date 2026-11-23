//! Ruff's per-file cache: `ruff check` keeps the files it found without diagnostics, and
//! `ruff format` the files it found formatted, so that a later run skips them.
//!
//! Invariants:
//! - A run with the cache prints, exits and writes what the same run without it would, apart from
//!   warnings logged while checking a file found clean (such as an invalid `# noqa` directive),
//!   which, as before, are printed only by the run that checks it.
//! - A file's result is used only under a key covering the build, the command, the settings the
//!   file is checked or formatted with, its permissions (which `ruff check`'s `EXE` rules read),
//!   and everything else the result depends on besides the file's path and content (for
//!   `ruff check`, its package root), and only while the answers to the questions checking the
//!   package's files asked of the file system (`isort`'s, finding first-party modules) are the
//!   same. Those are kept once per package and rechecked once per run: they're the same for most
//!   files, and change only when a module appears or disappears where `isort` looks, when every
//!   result of the package is dropped.
//! - A file's `stat` is trusted only outside its racy window; a file whose `stat` changed is read
//!   and keeps its result if its content didn't change.
//! - The cache never changes an answer by failing: anything unreadable is a miss, and anything
//!   unwritable is skipped.
//!
//! Each cache directory keeps, under the running version:
//! - `check/<package>` and `format/<package>`: the results of each package's files, by path
//!   relative to the package root;
//! - `store`: every result by key, path and content, shared by all of them;
//! - `builds`: the digests of the executables that ran;
//! - `walks/<run shape>`: what a search for project files found and read (see [`walk`]).

use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use itertools::Itertools;
use log::{debug, error};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use ruff_cache::{CacheKey, CacheKeyHasher};
use ruff_file_cache::{BuildId, Cache, CacheDir, Digest, Question, Retention, Stat, Timestamp};
use ruff_linter::VERSION;
use ruff_linter::package::PackageRoot;
use ruff_linter::probes::{self, Probe};
use ruff_linter::source_kind::{SourceError, SourceKind};
use ruff_python_ast::SourceType;
use ruff_workspace::Settings;
use ruff_workspace::resolver::Resolver;

pub(crate) mod walk;

/// How long a file's result is kept when no run looks it up: runs given some of a package's
/// files (as pre-commit hooks and editors do) keep the results of the others.
const KEEP_UNVISITED: Duration = Duration::from_hours(720); // 30 days.

/// What a command keeps for a file.
pub(crate) trait Verdict: Clone + Serialize + DeserializeOwned + Send + Sync {
    /// The command's name: its results are kept apart from other commands'.
    const COMMAND: &'static str;
}

/// `ruff check` found no diagnostics in the file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Clean;

impl Verdict for Clean {
    const COMMAND: &'static str = "check";
}

/// `ruff format` found the file formatted.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Formatted;

impl Verdict for Formatted {
    const COMMAND: &'static str = "format";
}

/// A cache directory a run opened, and the build that runs.
#[derive(Clone)]
struct OpenDir {
    /// `<cache-dir>/<version>`.
    dir: Arc<CacheDir>,
    /// The running build, as remembered in `dir`.
    build: BuildId,
}

impl OpenDir {
    /// Opens the cache directory `cache_dir`, initializing it. Returns `None` if it can't be
    /// created or the running build can't be identified: the run is then uncached.
    fn open(cache_dir: &Path) -> Option<Self> {
        if let Err(err) = Self::init(cache_dir) {
            error!(
                "Failed to initialize cache at {}: {err:?}",
                cache_dir.display()
            );
            return None;
        }
        let dir = CacheDir::open(&cache_dir.join(VERSION))?;
        let build = BuildId::current(&dir, "builds")?;
        Some(Self {
            dir: Arc::new(dir),
            build,
        })
    }

    /// Creates `path`, marked as a cache directory and ignored by Git.
    fn init(path: &Path) -> Result<()> {
        fs::create_dir_all(path)?;
        cachedir::ensure_tag(path)?;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join(".gitignore"))
        {
            Ok(mut file) => file.write_all(b"# Automatically created by ruff.\n*\n")?,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
            Err(err) => return Err(err.into()),
        }
        Ok(())
    }
}

/// The per-file results of one package, for one command.
struct PackageCache<R> {
    /// The package root: the cache's paths are relative to it.
    root: PathBuf,
    /// The running build.
    build: BuildId,
    /// The package's results.
    cache: Cache<R>,
}

/// The digests of the settings a run checks or formats files with, by the address of the
/// [`Settings`] the resolver holds: computed once per configuration rather than once per file.
struct SettingsKeys(FxHashMap<usize, u64>);

impl SettingsKeys {
    /// Hashes the part of each of `resolver`'s settings that `part` picks: what a command's
    /// results depend on.
    fn new(resolver: &Resolver, part: fn(&Settings) -> &dyn CacheKey) -> Self {
        Self(
            resolver
                .settings()
                .map(|settings| {
                    let mut hasher = CacheKeyHasher::new();
                    part(settings).cache_key(&mut hasher);
                    (Self::address(settings), hasher.finish())
                })
                .collect(),
        )
    }

    /// Returns the digest of `settings`, which the resolver holds.
    fn get(&self, settings: &Settings) -> Option<u64> {
        self.0.get(&Self::address(settings)).copied()
    }

    /// Returns where `settings` is: settings stay put in the resolver for the whole run.
    fn address(settings: &Settings) -> usize {
        std::ptr::from_ref(settings).addr()
    }
}

/// The per-file results of a run of one command, by package root.
pub(crate) struct PackageCaches<'a, R> {
    /// The caches, by the package root (or directory, for a file outside any package) they hold
    /// the results of.
    packages: FxHashMap<&'a Path, PackageCache<R>>,
    /// The cache directories they're in: their stores are committed once every cache is saved.
    dirs: Vec<Arc<CacheDir>>,
    /// The digests of the settings files are checked or formatted with.
    settings: SettingsKeys,
}

impl<'a, R: Verdict> PackageCaches<'a, R> {
    /// Opens the caches of `package_roots`, in the cache directory of each one's settings, for a
    /// run that started at `started`. `part` picks what of a file's settings its result depends
    /// on.
    pub(crate) fn open(
        package_roots: &FxHashMap<&'a Path, Option<PackageRoot<'a>>>,
        resolver: &Resolver,
        started: Timestamp,
        part: fn(&Settings) -> &dyn CacheKey,
    ) -> Self {
        let dirs: FxHashMap<&Path, Option<OpenDir>> = resolver
            .settings()
            .map(|settings| settings.cache_dir.as_path())
            .unique()
            .map(|cache_dir| (cache_dir, OpenDir::open(cache_dir)))
            .collect();
        let packages = package_roots
            .iter()
            .map(|(package, package_root)| package_root.map(PackageRoot::path).unwrap_or(package))
            .unique()
            .par_bridge()
            .filter_map(|root| {
                let open = dirs
                    .get(resolver.resolve(root).cache_dir.as_path())?
                    .as_ref()?;
                let name = format!("{}/{}", R::COMMAND, Digest::of(&root));
                let cache = Cache::open(Arc::clone(&open.dir), name, started)
                    .keep_unvisited_for(KEEP_UNVISITED);
                let package = PackageCache {
                    root: root.to_path_buf(),
                    build: open.build,
                    cache,
                };
                Some((root, package))
            })
            .collect();
        Self {
            packages,
            dirs: dirs.into_values().flatten().map(|open| open.dir).collect(),
            settings: SettingsKeys::new(resolver, part),
        }
    }

    /// Returns the file at `path`, in the package rooted at `package_root`, checked or formatted
    /// with `settings`, whose result also depends on `extra`. Returns `None` if it isn't cached:
    /// it's outside the package, it's gone, or its path isn't UTF-8.
    pub(crate) fn file(
        &self,
        package_root: &Path,
        path: &Path,
        settings: &Settings,
        extra: impl Hash,
    ) -> Option<CachedFile<'_, R>> {
        let Some(package) = self.packages.get(package_root) else {
            debug!("No cache found for {}", package_root.display());
            return None;
        };
        let relative = path.strip_prefix(&package.root).ok()?;
        let stat = Stat::of_target(path)?;
        let settings = self.settings.get(settings)?;
        let generation = package.cache.generation();
        let key = Digest::of(&(
            package.build,
            R::COMMAND,
            settings,
            stat.mode(),
            extra,
            generation,
        ));
        Some(CachedFile {
            cache: &package.cache,
            path: relative.to_path_buf(),
            key,
            stat,
        })
    }

    /// Saves every package's results, and commits what they used and added to their stores.
    pub(crate) fn persist(self) {
        self.packages
            .into_par_iter()
            .for_each(|(_, package)| package.cache.save());
        for dir in self.dirs {
            dir.commit(Retention::default());
        }
    }
}

/// A file a command is about to read, and its cached result.
pub(crate) struct CachedFile<'a, R> {
    /// The package's results.
    cache: &'a Cache<R>,
    /// The file's path, relative to the package root.
    path: PathBuf,
    /// Everything the file's result depends on besides its path and content.
    key: Digest,
    /// The file's `stat`, taken before it's read.
    stat: Stat,
}

impl<R: Verdict> CachedFile<'_, R> {
    /// Returns the file's result if its `stat` says it's unchanged: then it needn't be read.
    pub(crate) fn by_stat(&self) -> Option<R> {
        self.cache.get_by_stat(&self.path, self.key, self.stat)
    }

    /// Returns the file's result for `content`, its digest once read.
    pub(crate) fn by_content(&self, content: Digest) -> Option<R> {
        self.cache
            .get_by_content(&self.path, self.key, self.stat, content)
    }

    /// Puts `result` for the file, read as `content`.
    pub(crate) fn put(&self, content: Digest, result: R) {
        self.cache
            .put(&self.path, self.key, self.stat, content, result);
    }

    /// Puts `result` for the file after the command wrote `content` to it.
    pub(crate) fn put_written(&self, content: Digest, result: R, absolute: &Path) {
        if let Some(stat) = Stat::of_target(absolute) {
            self.cache.put(&self.path, self.key, stat, content, result);
        }
    }
}

impl CachedFile<'_, Clean> {
    /// Puts the file as clean, read as `content`, while `probes`, the questions checking it asked
    /// of the file system, get the same answers.
    pub(crate) fn put_clean(&self, content: Digest, probes: &[Probe]) {
        let shared = self.cache.shared();
        for probe in probes {
            let question = match probe.question {
                probes::Question::IsDir => Question::IsDir,
                probes::Question::IsFile => Question::IsFile,
            };
            shared.answer(&probe.path, question, probe.answer);
        }
        self.put(content, Clean);
    }
}

/// A source file's bytes, read once to hash them and, on a miss, to check or format them.
pub(crate) struct SourceBytes {
    /// The file's content.
    bytes: Vec<u8>,
    /// Its digest.
    digest: Digest,
}

impl SourceBytes {
    /// Reads the file at `path`.
    pub(crate) fn read(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let digest = Digest::of_bytes(&bytes);
        Ok(Self { bytes, digest })
    }

    /// Returns the digest of the content.
    pub(crate) fn digest(&self) -> Digest {
        self.digest
    }

    /// Returns the source the content holds, as [`SourceKind::from_path`] reads it. A content
    /// it can't read is read again with it, so that the error is the one an uncached run reports.
    pub(crate) fn into_source_kind(
        self,
        path: &Path,
        source_type: SourceType,
    ) -> Result<Option<SourceKind>, SourceError> {
        String::from_utf8(self.bytes)
            .ok()
            .and_then(|source| SourceKind::from_source_code(source, source_type).ok())
            .map_or_else(|| SourceKind::from_path(path, source_type), Ok)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use filetime::{FileTime, set_file_mtime};
    use test_case::test_case;

    use ruff_file_cache::Timestamp;
    use ruff_linter::package::PackageRoot;
    use ruff_linter::registry::Rule;
    use ruff_linter::settings::LinterSettings;
    use ruff_linter::settings::flags;
    use ruff_linter::settings::types::UnsafeFixes;
    use ruff_python_ast::{PySourceType, SourceType};
    use ruff_workspace::Settings;
    use ruff_workspace::resolver::{
        PyprojectDiscoveryStrategy, ResolvedSettings, Resolver, RootSettings,
    };
    use rustc_hash::FxHashMap;

    use super::{Clean, Formatted, PackageCaches};
    use crate::commands::format::{FormatMode, FormatResult, format_path};
    use crate::diagnostics::{Diagnostics, lint_path};

    /// A package of files, linted and formatted with a cache.
    struct TestPackage {
        _tmp: tempfile::TempDir,
        root: PathBuf,
        root_settings: RootSettings,
    }

    impl TestPackage {
        fn new(rules: Vec<Rule>) -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let root = dunce::canonicalize(tmp.path()).unwrap().join("package");
            fs::create_dir_all(&root).unwrap();
            let settings = Settings {
                cache_dir: tmp.path().join(".ruff_cache"),
                linter: LinterSettings::for_rules(rules),
                ..Settings::default()
            };
            Self {
                _tmp: tmp,
                root,
                root_settings: RootSettings {
                    strategy: PyprojectDiscoveryStrategy::Fixed,
                    resolved: ResolvedSettings::new(settings),
                },
            }
        }

        fn write(&self, name: &str, source: &str) -> PathBuf {
            let path = self.root.join(name);
            fs::write(&path, source).unwrap();
            path
        }

        /// When a run starts that trusts every `stat` taken before the test's writes.
        fn later() -> Timestamp {
            Timestamp::now() + Duration::from_secs(10)
        }

        fn package_roots(&self) -> FxHashMap<&Path, Option<PackageRoot<'_>>> {
            FxHashMap::from_iter([(self.root.as_path(), Some(PackageRoot::root(&self.root)))])
        }

        /// Lints `paths` with the cache, as a run started at `started`.
        fn lint(&self, paths: &[&Path], started: Timestamp) -> Diagnostics {
            let resolver = Resolver::new(&self.root_settings);
            let package_roots = self.package_roots();
            let caches =
                PackageCaches::<Clean>::open(&package_roots, &resolver, started, |s| &s.linter);
            let mut diagnostics = Diagnostics::default();
            for path in paths {
                let package = Some(PackageRoot::root(&self.root));
                let settings = resolver.resolve(path);
                diagnostics += lint_path(
                    path,
                    package,
                    &settings.linter,
                    caches.file(&self.root, path, settings, package),
                    flags::Noqa::Enabled,
                    flags::FixMode::Generate,
                    UnsafeFixes::Enabled,
                )
                .unwrap();
            }
            caches.persist();
            diagnostics
        }

        /// Returns whether the cache knows `path` is clean from its `stat` alone.
        fn known_clean_by_stat(&self, path: &Path) -> bool {
            let resolver = Resolver::new(&self.root_settings);
            let package_roots = self.package_roots();
            let caches =
                PackageCaches::<Clean>::open(&package_roots, &resolver, Self::later(), |s| {
                    &s.linter
                });
            let package = Some(PackageRoot::root(&self.root));
            caches
                .file(&self.root, path, resolver.resolve(path), package)
                .is_some_and(|file| file.by_stat().is_some())
        }

        /// Formats `path` with the cache, as a run started at `started`.
        fn format(&self, path: &Path, mode: FormatMode, started: Timestamp) -> FormatResult {
            let resolver = Resolver::new(&self.root_settings);
            let package_roots = self.package_roots();
            let caches =
                PackageCaches::<Formatted>::open(&package_roots, &resolver, started, |s| {
                    &s.formatter
                });
            let settings = resolver.resolve(path);
            let result = format_path(
                path,
                &settings.formatter,
                SourceType::Python(PySourceType::Python),
                mode,
                None,
                caches.file(&self.root, path, settings, ()),
            )
            .unwrap();
            caches.persist();
            result
        }
    }

    #[test]
    fn a_clean_file_is_checked_again_when_isort_would_find_a_module_elsewhere() {
        let mut package = TestPackage::new(vec![Rule::UnsortedImports]);
        let src = package.root.join("src");
        fs::create_dir_all(&src).unwrap();
        let settings = &mut package.root_settings.resolved.settings;
        settings.linter.src = vec![src.clone()];
        // Both third-party, and sorted: until `foo` is first-party, which goes in its own section.
        let path = package.write("main.py", "import bar\nimport foo\n");

        assert!(
            package
                .lint(&[&path], TestPackage::later())
                .inner
                .is_empty()
        );
        assert!(package.known_clean_by_stat(&path));

        fs::create_dir(src.join("foo")).unwrap();
        assert!(!package.known_clean_by_stat(&path));
        let diagnostics = package.lint(&[&path], TestPackage::later());
        assert_eq!(diagnostics.inner.len(), 1, "{diagnostics:?}");

        fs::remove_dir(src.join("foo")).unwrap();
        fs::write(src.join("foo.py"), "").unwrap();
        assert_eq!(package.lint(&[&path], TestPackage::later()).inner.len(), 1);
        fs::remove_file(src.join("foo.py")).unwrap();
        assert!(
            package
                .lint(&[&path], TestPackage::later())
                .inner
                .is_empty()
        );
        assert!(package.known_clean_by_stat(&path));
    }

    #[test_case("../ruff_linter/resources/test/fixtures/pyflakes"; "pyflakes_fixtures")]
    #[test_case("../ruff_notebook/resources/test/fixtures/jupyter"; "notebook_fixtures")]
    fn a_cached_run_finds_what_an_uncached_run_finds(fixtures: &str) {
        let package = TestPackage::new(vec![Rule::UnusedVariable, Rule::UnusedImport]);
        let mut paths = Vec::new();
        for entry in fs::read_dir(fixtures).unwrap() {
            let path = entry.unwrap().path();
            let extension = path.extension().and_then(|extension| extension.to_str());
            if path.is_file()
                && matches!(extension, Some("py" | "ipynb"))
                && let Some(name) = path.file_name()
            {
                fs::copy(&path, package.root.join(name)).unwrap();
                paths.push(package.root.join(name));
            }
        }
        paths.sort();
        assert!(!paths.is_empty(), "no files checked");

        let started = TestPackage::later();
        let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let uncached = package.lint(&paths, started);
        let cached = package.lint(&paths, started);
        assert_eq!(uncached.inner, cached.inner);
    }

    #[test]
    fn a_clean_file_is_skipped_and_one_with_diagnostics_is_not() {
        let package = TestPackage::new(vec![Rule::UnusedImport]);
        let clean = package.write("clean.py", "a = 1\n");
        let unclean = package.write("unclean.py", "import os\n");
        let started = TestPackage::later();
        assert_eq!(package.lint(&[&clean, &unclean], started).inner.len(), 1);
        assert!(package.known_clean_by_stat(&clean));
        assert!(!package.known_clean_by_stat(&unclean));
        assert!(package.lint(&[&clean], started).inner.is_empty());
        assert_eq!(package.lint(&[&unclean], started).inner.len(), 1);
    }

    #[test]
    fn an_edit_that_puts_the_mtime_back_is_seen() {
        let package = TestPackage::new(vec![Rule::UnusedImport]);
        let path = package.write("source.py", "x = 12345\n");
        let started = TestPackage::later();
        assert!(package.lint(&[&path], started).inner.is_empty());

        let mtime = FileTime::from_last_modification_time(&fs::metadata(&path).unwrap());
        fs::write(&path, "import os\n").unwrap();
        set_file_mtime(&path, mtime).unwrap();
        // The edit is the same size too: only the inode change time tells.
        assert!(!package.known_clean_by_stat(&path));
        let started = TestPackage::later();
        assert_eq!(package.lint(&[&path], started).inner.len(), 1);
    }

    #[test]
    fn a_touched_file_keeps_its_result_by_content() {
        let package = TestPackage::new(vec![Rule::UnusedImport]);
        let path = package.write("source.py", "a = 1\n");
        let started = TestPackage::later();
        assert!(package.lint(&[&path], started).inner.is_empty());
        fs::write(&path, "a = 1\n").unwrap();
        assert!(!package.known_clean_by_stat(&path));
        assert!(
            package
                .lint(&[&path], TestPackage::later())
                .inner
                .is_empty()
        );
        assert!(
            package.known_clean_by_stat(&path),
            "kept again by its new stat"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_permission_change_is_seen() {
        use std::os::unix::fs::PermissionsExt;
        let package = TestPackage::new(vec![Rule::NonExecutableFileWithShebang]);
        let path = package.write("script.py", "#!/usr/bin/env python\n");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        let started = TestPackage::later();
        if !package.lint(&[&path], started).inner.is_empty() {
            // The file system can't tell executable files (as under WSL): nothing to check.
            return;
        }
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(package.lint(&[&path], TestPackage::later()).inner.len(), 1);
    }

    #[test]
    fn a_formatted_file_is_skipped_and_a_written_one_kept_by_content() {
        let package = TestPackage::new(vec![]);
        let path = package.write("source.py", "a  =  1\n");
        let started = TestPackage::later();
        assert!(matches!(
            package.format(&path, FormatMode::Check, started),
            FormatResult::Diff { .. }
        ));
        assert!(matches!(
            package.format(&path, FormatMode::Write, started),
            FormatResult::Formatted
        ));
        assert_eq!(fs::read_to_string(&path).unwrap(), "a = 1\n");
        assert!(matches!(
            package.format(&path, FormatMode::Check, TestPackage::later()),
            FormatResult::Unchanged
        ));
        fs::write(&path, "a  =  2\n").unwrap();
        assert!(matches!(
            package.format(&path, FormatMode::Check, TestPackage::later()),
            FormatResult::Diff { .. }
        ));
    }
}
