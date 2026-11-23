//! The walk cache: what a search for project files found, and everything it read, so that a run
//! whose inputs all read the same skips walking the file system.
//!
//! A walk reads directory listings, configuration files (and the files they extend, and the
//! `pyproject.toml` files a `requires-python` fallback looks for), and the ignore files the
//! `ignore` crate reads: `.gitignore` and `.ignore` in each listed directory and above each path
//! argument, Git's `info/exclude`, and the global excludes file. A directory's listing is
//! recorded by its `stat`, which changes when an entry is added, removed or renamed; a file by its
//! content; a file looked for and missing as missing.
//!
//! A cache directory's contents are the cache's: a walk that lists them (when `exclude` drops
//! the default `.ruff_cache` and no Git repository applies its `.gitignore`) doesn't record them,
//! or every run's writes would undo the walk it keeps.
//!
//! The walk is kept per run shape: what decides what it finds, besides what it reads. `ruff check`
//! and `ruff format` search the same way, so they share it. A kept walk's configurations are
//! resolved again, as the files they come from read the same, and the walk is used only if each
//! resolves to the settings it found.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};

use anyhow::Result;
use log::debug;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use ruff_cache::{CACHE_DIR_NAME, CacheKey, CacheKeyHasher};
use ruff_file_cache::{BuildId, CacheDir, Digest, Inputs, Recorder, Timestamp};
use ruff_linter::fs::normalize_path;
use ruff_workspace::Settings;
use ruff_workspace::pyproject::DiscoveryObserver;
use ruff_workspace::resolver::{
    ConfigurationTransformer, PyprojectDiscoveryStrategy, ResolvedFile, Resolver, RootSettings,
    project_files_in_path,
};

use super::OpenDir;

/// Run shapes whose walks are kept, by most recent write.
const MAX_WALKS: usize = 256;

/// What decides what a walk finds, besides what it reads: runs of one shape share a walk.
#[derive(Hash)]
struct RunShape<'a> {
    /// The running build.
    build: BuildId,
    /// The current directory: relative path arguments and the global excludes file are resolved
    /// against it.
    cwd: &'a Path,
    /// The path arguments.
    paths: &'a [PathBuf],
    /// Whether closer configurations apply to the files under them.
    hierarchical: bool,
    /// The root settings, which decide what's included and excluded unless a closer
    /// configuration does.
    root_settings: u64,
    /// The environment variables that decide where Git's global configuration and excludes are.
    git_environment: Vec<Option<OsString>>,
}

/// A configuration a walk found and the directory it applies to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FoundConfiguration {
    /// The directory it was found for.
    directory: PathBuf,
    /// The configuration file.
    path: PathBuf,
    /// Its settings' cache key: a kept walk is used only if the configuration resolves to the
    /// same settings again (the command-line overrides may differ).
    settings: u64,
}

/// How a file came to be checked or formatted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Origin {
    /// Given on the command line.
    Given,
    /// Found under a directory given on the command line.
    Found,
}

/// A file a walk found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct WalkedFile {
    /// Its path, as the walk found it.
    path: PathBuf,
    /// How it came to be checked or formatted.
    origin: Origin,
}

impl WalkedFile {
    /// Returns the file as the walk returns it.
    fn resolved(&self) -> ResolvedFile {
        match self.origin {
            Origin::Given => ResolvedFile::Root(self.path.clone()),
            Origin::Found => ResolvedFile::Nested(self.path.clone()),
        }
    }
}

/// What a walk found and read: a later run whose inputs read the same uses it instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Walk {
    /// What the walk read.
    inputs: Inputs,
    /// The configurations it found, in the order it added them to the resolver.
    configurations: Vec<FoundConfiguration>,
    /// The files it found, in the order it returned them.
    files: Vec<WalkedFile>,
}

impl Walk {
    /// Returns what `files` and `resolver` hold, read as `inputs`, or `None` if the walk reported
    /// an error (which a kept walk wouldn't report again).
    fn new(
        inputs: Inputs,
        files: &[Result<ResolvedFile, ignore::Error>],
        resolver: &Resolver,
    ) -> Option<Self> {
        let files = files
            .iter()
            .map(|file| match file.as_ref().ok()? {
                ResolvedFile::Root(path) => Some(WalkedFile {
                    path: path.clone(),
                    origin: Origin::Given,
                }),
                ResolvedFile::Nested(path) => Some(WalkedFile {
                    path: path.clone(),
                    origin: Origin::Found,
                }),
            })
            .collect::<Option<Vec<_>>>()?;
        let configurations = resolver
            .configurations()
            .map(|(directory, resolved)| {
                Some(FoundConfiguration {
                    directory: directory.to_path_buf(),
                    path: resolved.config_path()?.to_path_buf(),
                    settings: WalkCache::settings_key(&resolved.settings),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            inputs,
            configurations,
            files,
        })
    }

    /// Returns the resolver the walk built, with its configurations resolved again, or `None` if
    /// one fails to resolve or resolves to other settings.
    fn resolver<'a>(
        &self,
        root_settings: &'a RootSettings,
        transformer: &dyn ConfigurationTransformer,
    ) -> Option<Resolver<'a>> {
        let mut resolver = Resolver::new(root_settings);
        let found = self
            .configurations
            .iter()
            .map(|found| (found.directory.as_path(), found.path.as_path()));
        resolver.add_configurations(found, transformer).ok()?;
        let same = resolver
            .configurations()
            .map(|(_, resolved)| WalkCache::settings_key(&resolved.settings))
            .eq(self.configurations.iter().map(|found| found.settings));
        if !same {
            debug!("Not reusing the last walk: its configurations resolve to other settings");
        }
        same.then_some(resolver)
    }
}

/// The walk cache of one run.
pub(crate) struct WalkCache {
    /// The root settings' cache directory.
    dir: Arc<CacheDir>,
    /// Where this run shape's walk is kept in `dir`.
    name: String,
    /// When the run started, before anything was read.
    started: Timestamp,
    /// The last walk of this run shape.
    last: Option<Walk>,
}

impl WalkCache {
    /// Opens the walk cache for a search of `paths` with `root_settings`, in a run that started at
    /// `started`. Returns `None` if the cache can't be opened: the run then walks, uncached.
    pub(crate) fn open(
        paths: &[PathBuf],
        root_settings: &RootSettings,
        started: Timestamp,
    ) -> Option<Self> {
        let open = OpenDir::open(&root_settings.resolved.settings.cache_dir)?;
        let shape = RunShape {
            build: open.build,
            cwd: &env::current_dir().ok()?,
            paths,
            hierarchical: matches!(
                root_settings.strategy,
                PyprojectDiscoveryStrategy::Hierarchical
            ),
            root_settings: Self::settings_key(&root_settings.resolved.settings),
            git_environment: GitFiles::ENVIRONMENT.iter().map(env::var_os).collect(),
        };
        let name = format!("walks/{}", Digest::of(&shape));
        let last = open.dir.read(&name);
        Some(Self {
            dir: open.dir,
            name,
            started,
            last,
        })
    }

    /// Finds the project files in `paths`, as [`project_files_in_path`] does: from the last walk,
    /// if everything it read reads the same, or by walking, recording what's read for the next
    /// run.
    pub(crate) fn project_files<'a>(
        self,
        paths: &[PathBuf],
        root_settings: &'a RootSettings,
        transformer: &(dyn ConfigurationTransformer + Sync),
    ) -> Result<(Vec<Result<ResolvedFile, ignore::Error>>, Resolver<'a>)> {
        if let Some((walk, resolver)) = self.replay(root_settings, transformer) {
            debug!("Reusing the last walk: nothing it read changed");
            let files = walk.files.iter().map(|file| Ok(file.resolved())).collect();
            if self.last.as_ref() != Some(&walk) {
                self.dir.write(&self.name, &walk);
            }
            return Ok((files, resolver));
        }

        let recorder = WalkRecorder::new(self.started, paths, root_settings);
        let (files, resolver) =
            project_files_in_path(paths, root_settings, transformer, Some(&recorder))?;
        if let Some(walk) = recorder
            .finish()
            .and_then(|inputs| Walk::new(inputs, &files, &resolver))
        {
            self.dir.write(&self.name, &walk);
            self.dir.prune("walks", MAX_WALKS);
        }
        Ok((files, resolver))
    }

    /// Returns the last walk, with its inputs as they read now, and its resolver, if everything
    /// it read reads the same and its configurations resolve to the same settings.
    fn replay<'a>(
        &self,
        root_settings: &'a RootSettings,
        transformer: &dyn ConfigurationTransformer,
    ) -> Option<(Walk, Resolver<'a>)> {
        let last = self.last.as_ref()?;
        let inputs = last.inputs.recheck(self.started)?;
        let resolver = last.resolver(root_settings, transformer)?;
        Some((
            Walk {
                inputs,
                ..last.clone()
            },
            resolver,
        ))
    }

    /// Returns the cache key of `settings`.
    fn settings_key(settings: &Settings) -> u64 {
        let mut hasher = CacheKeyHasher::new();
        settings.cache_key(&mut hasher);
        std::hash::Hasher::finish(&hasher)
    }
}

/// Records what a walk reads, from any of its threads: what settings discovery reports, and
/// what the `ignore` crate reads as it lists the same directories.
struct WalkRecorder {
    /// Where what the walk reads is recorded.
    recorder: Recorder,
    /// Whether the walk reads ignore files.
    respect_gitignore: bool,
    /// The directories the walk listed: a file looked for and missing in one needn't be recorded,
    /// as its appearance changes the directory's listing.
    listed: Listed,
    /// The root settings' cache directory, whose contents aren't recorded.
    cache_dir: PathBuf,
}

impl WalkRecorder {
    /// Returns a recorder for a walk of `paths` with `root_settings` that started at `started`,
    /// having recorded what the walk reads before it lists anything: the path arguments, and the
    /// ignore files above them and in Git's global configuration.
    fn new(started: Timestamp, paths: &[PathBuf], root_settings: &RootSettings) -> Self {
        let walk = Self {
            recorder: Recorder::new(started),
            respect_gitignore: root_settings
                .resolved
                .settings
                .file_resolver
                .respect_gitignore,
            listed: Listed::default(),
            // Absolute, as the walk's paths are: `--cache-dir` may be relative.
            cache_dir: normalize_path(&root_settings.resolved.settings.cache_dir),
        };
        for path in paths {
            let path = normalize_path(path);
            walk.recorder.kind(&path);
            if walk.respect_gitignore
                && let Ok(absolute) = fs::canonicalize(&path)
            {
                for parent in absolute.ancestors().skip(1) {
                    GitFiles::new(&walk.recorder).parent(parent);
                }
            }
        }
        if walk.respect_gitignore {
            GitFiles::new(&walk.recorder).global();
        }
        walk
    }

    /// Returns whether `path` is in a cache directory: the root settings', or another a nested
    /// configuration may use, by its default name and tag.
    fn in_cache_dir(&self, path: &Path) -> bool {
        path.ancestors().any(|dir| {
            dir == self.cache_dir
                || (dir.file_name() == Some(CACHE_DIR_NAME.as_ref())
                    && dir.join("CACHEDIR.TAG").is_file())
        })
    }

    /// Returns what the walk read, or `None` if something couldn't be recorded.
    fn finish(self) -> Option<Inputs> {
        self.recorder.finish()
    }
}

impl DiscoveryObserver for WalkRecorder {
    fn listing(&self, path: &Path) {
        if self.in_cache_dir(path) {
            return;
        }
        self.recorder.dir(path);
        self.listed.insert(path);
        if self.respect_gitignore {
            GitFiles::new(&self.recorder).listed(path);
        }
    }

    fn reading(&self, path: &Path) {
        if path
            .parent()
            .is_some_and(|parent| self.listed.contains(parent))
        {
            self.recorder.file_if_present(path);
        } else {
            self.recorder.file(path);
        }
    }
}

/// The files the `ignore` crate reads to decide what's ignored, as of `ignore` 0.4.33: in each
/// directory it lists or that's above a path argument, `.ignore`, `.gitignore`, and whether
/// there's a `.git` or `.jj`; for a Git repository, its `info/exclude`; and the global excludes
/// file Git's configuration names.
struct GitFiles<'a> {
    /// Where they're recorded.
    recorder: &'a Recorder,
}

impl<'a> GitFiles<'a> {
    /// The environment variables that decide where Git's configuration and global excludes are.
    const ENVIRONMENT: [&'static str; 4] = [
        "HOME",
        "XDG_CONFIG_HOME",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
    ];

    /// Records into `recorder`.
    fn new(recorder: &'a Recorder) -> Self {
        Self { recorder }
    }

    /// Records the files read in `dir`, which the walk lists: any of them appearing changes its
    /// listing, so only those present are recorded.
    fn listed(&self, dir: &Path) {
        for name in [".ignore", ".gitignore"] {
            self.recorder.file_if_present(&dir.join(name));
        }
        if fs::symlink_metadata(dir.join(".git")).is_ok() {
            self.repository(dir);
        }
        if fs::symlink_metadata(dir.join(".jj")).is_ok() {
            self.recorder.kind(&dir.join(".jj"));
        }
    }

    /// Records the files read in `dir`, above a path argument: each, present or not.
    fn parent(&self, dir: &Path) {
        for name in [".ignore", ".gitignore"] {
            self.recorder.file(&dir.join(name));
        }
        self.recorder.kind(&dir.join(".jj"));
        if fs::symlink_metadata(dir.join(".git")).is_ok() {
            self.repository(dir);
        } else {
            self.recorder.file(&dir.join(".git"));
        }
    }

    /// Records Git's files for a repository rooted at `dir`: `.git` (a directory, or a worktree's
    /// file and the `commondir` it points to) and `info/exclude`, as the `ignore` crate finds them.
    fn repository(&self, dir: &Path) {
        let dot_git = dir.join(".git");
        self.recorder.file(&dot_git);
        if fs::symlink_metadata(&dot_git).is_ok_and(|meta| meta.is_file()) {
            if let Some(common) = self.common_dir(&dot_git) {
                self.recorder.file(&common.join("info/exclude"));
            }
        } else {
            self.recorder.file(&dot_git.join("info/exclude"));
        }
    }

    /// Returns the common Git directory a worktree's `.git` file points to, as the `ignore` crate
    /// resolves it: relative paths against the process's working directory.
    fn common_dir(&self, dot_git: &Path) -> Option<PathBuf> {
        let line = Self::first_line(dot_git)?;
        let git_dir = PathBuf::from(line.strip_prefix("gitdir: ")?);
        let commondir_file = git_dir.join("commondir");
        self.recorder.file(&commondir_file);
        let commondir = Self::first_line(&commondir_file)?;
        Some(if commondir.starts_with('.') {
            git_dir.join(commondir)
        } else {
            PathBuf::from(commondir)
        })
    }

    /// Records Git's global configuration files and the global excludes file they name.
    fn global(&self) {
        let home = env::home_dir();
        let config_home = env::var_os("XDG_CONFIG_HOME")
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|home| home.join(".config")));
        let candidates = [
            env::var_os("GIT_CONFIG_GLOBAL")
                .filter(|path| !path.is_empty())
                .map(PathBuf::from),
            home.map(|home| home.join(".gitconfig")),
            config_home.map(|dir| dir.join("git/config")),
            Some(
                env::var_os("GIT_CONFIG_SYSTEM")
                    .filter(|path| !path.is_empty())
                    .map_or_else(|| PathBuf::from("/etc/gitconfig"), PathBuf::from),
            ),
            ignore::gitignore::gitconfig_excludes_path(),
        ];
        for path in candidates.iter().flatten() {
            self.recorder.file(path);
        }
    }

    /// Returns the first line of the file at `path`.
    fn first_line(path: &Path) -> Option<String> {
        io::BufReader::new(fs::File::open(path).ok()?)
            .lines()
            .next()?
            .ok()
    }
}

/// The directories a walk listed, shared by its threads.
#[derive(Default)]
struct Listed(RwLock<FxHashSet<PathBuf>>);

impl Listed {
    /// Adds `dir`.
    fn insert(&self, dir: &Path) {
        self.0
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(dir.to_path_buf());
    }

    /// Returns whether `dir` was listed.
    fn contains(&self, dir: &Path) -> bool {
        self.0
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(dir)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};

    use ruff_file_cache::Timestamp;
    use ruff_workspace::configuration::Configuration;
    use ruff_workspace::resolver::{
        PyprojectDiscoveryStrategy, ResolvedFile, ResolvedSettings, RootSettings,
        project_files_in_path,
    };

    use super::{WalkCache, WalkRecorder};
    use crate::args::ConfigArguments;

    const DIRS: [&str; 4] = ["", "a", "a/b", "c"];
    const FILES: [(&str, &[&str]); 5] = [
        ("x.py", &["x = 1\n"]),
        ("y.pyi", &["y: int\n"]),
        (".gitignore", &["", "x.py\n", "b/\n", "*.pyi\n"]),
        (
            "ruff.toml",
            &[
                "",
                "extend-exclude = [\"x.py\"]\n",
                "extend-include = [\"*.pyw\"]\n",
                "extend = \"../base.toml\"\n",
            ],
        ),
        ("z.pyw", &["z = 1\n"]),
    ];
    const BASES: [&str; 2] = ["", "extend-exclude = [\"y.pyi\"]\n"];

    /// A change to the project after a run.
    #[derive(Debug, Clone)]
    enum Change {
        Write {
            dir: usize,
            file: usize,
            content: usize,
        },
        Remove {
            dir: usize,
            file: usize,
        },
        /// Edit the configurations `extend = "../base.toml"` points to.
        WriteBase(usize),
        /// Edit Git's `info/exclude`.
        WriteExclude(bool),
        /// Remove a directory and everything in it.
        RemoveDir(usize),
        /// Write or remove a configuration above the project, which applies to it when it has
        /// none of its own.
        Above(Option<usize>),
    }

    impl Arbitrary for Change {
        fn arbitrary(g: &mut Gen) -> Self {
            let mut pick = |n: usize| usize::arbitrary(g) % n;
            let (dir, file) = (pick(DIRS.len()), pick(FILES.len()));
            let content = pick(4);
            match pick(10) {
                0..=4 => Self::Write { dir, file, content },
                5 | 6 => Self::Remove { dir, file },
                7 => Self::WriteBase(pick(BASES.len())),
                8 => Self::WriteExclude(pick(2) == 0),
                9 if pick(2) == 0 => Self::Above((pick(2) == 0).then_some(content % 3)),
                _ => Self::RemoveDir(1 + pick(DIRS.len() - 1)),
            }
        }
    }

    impl Change {
        fn apply(&self, root: &Path) {
            match *self {
                Self::Write { dir, file, content } => {
                    let (name, contents) = FILES[file];
                    let dir = root.join("project").join(DIRS[dir]);
                    fs::create_dir_all(&dir).unwrap();
                    fs::write(dir.join(name), contents[content % contents.len()]).unwrap();
                }
                Self::Remove { dir, file } => {
                    let _ =
                        fs::remove_file(root.join("project").join(DIRS[dir]).join(FILES[file].0));
                }
                Self::WriteBase(content) => {
                    // Where `extend = "../base.toml"` points from each directory.
                    for dir in ["", "project", "project/a"] {
                        let _ = fs::write(root.join(dir).join("base.toml"), BASES[content]);
                    }
                }
                Self::WriteExclude(yes) => {
                    let contents = if yes { "c/\n" } else { "" };
                    fs::write(root.join("project/.git/info/exclude"), contents).unwrap();
                }
                Self::RemoveDir(dir) => {
                    let _ = fs::remove_dir_all(root.join("project").join(DIRS[dir]));
                }
                Self::Above(Some(content)) => {
                    fs::write(root.join("ruff.toml"), FILES[3].1[content]).unwrap();
                }
                Self::Above(None) => {
                    let _ = fs::remove_file(root.join("ruff.toml"));
                }
            }
        }
    }

    /// What a search finds: its files and the configurations it resolved.
    fn found(
        files: &[Result<ResolvedFile, ignore::Error>],
        resolver: &ruff_workspace::resolver::Resolver,
    ) -> (Vec<String>, Vec<(PathBuf, Option<PathBuf>)>) {
        let files = files
            .iter()
            .map(|file| match file {
                Ok(file) => format!("{file:?}"),
                Err(err) => format!("error: {err}"),
            })
            .collect();
        let configurations = resolver
            .configurations()
            .map(|(dir, resolved)| {
                (
                    dir.to_path_buf(),
                    resolved.config_path().map(Path::to_path_buf),
                )
            })
            .collect();
        (files, configurations)
    }

    /// For any project and any changes to it, a search with the walk cache finds what a search
    /// without it finds.
    #[test]
    fn a_cached_walk_finds_what_walking_finds() {
        #[expect(clippy::needless_pass_by_value, reason = "quickcheck passes values")]
        fn property(changes: Vec<Change>) -> TestResult {
            let tmp = tempfile::tempdir().unwrap();
            let root = dunce::canonicalize(tmp.path()).unwrap();
            let project = root.join("project");
            fs::create_dir_all(project.join(".git/info")).unwrap();
            fs::write(root.join("base.toml"), BASES[0]).unwrap();
            let settings = Configuration {
                cache_dir: Some(root.join("cache")),
                ..Configuration::default()
            }
            .into_settings(&project)
            .unwrap();
            let root_settings = RootSettings {
                strategy: PyprojectDiscoveryStrategy::Hierarchical,
                resolved: ResolvedSettings::new(settings),
            };
            let paths = [project];
            let transformer = ConfigArguments::default();
            for change in changes.iter().map(Some).chain([None]) {
                // Past the racy window, so that the `stat`s just taken can be trusted.
                let started = Timestamp::now() + Duration::from_secs(10);
                let Ok((files, resolver)) =
                    project_files_in_path(&paths, &root_settings, &transformer, None)
                else {
                    // An invalid configuration fails both searches alike.
                    if let Some(change) = change {
                        change.apply(&root);
                    }
                    continue;
                };
                let expected = found(&files, &resolver);
                for _ in 0..2 {
                    let cache = WalkCache::open(&paths, &root_settings, started).unwrap();
                    let (files, resolver) = cache
                        .project_files(&paths, &root_settings, &transformer)
                        .unwrap();
                    if found(&files, &resolver) != expected {
                        return TestResult::error(format!(
                            "cached {:?} != walked {expected:?}",
                            found(&files, &resolver)
                        ));
                    }
                }
                // File systems whose timestamps are coarse wouldn't tell a change in the same
                // tick from the state just recorded; the cache's racy window covers that in a
                // real run, but this test trusts every `stat` it takes.
                std::thread::sleep(Duration::from_millis(15));
                if let Some(change) = change {
                    change.apply(&root);
                }
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(30)
            .rng(Gen::new(12))
            .quickcheck(property as fn(Vec<Change>) -> TestResult);
    }

    #[test]
    fn a_walk_into_a_cache_directory_records_nothing_in_it() {
        use ruff_workspace::pyproject::DiscoveryObserver;

        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let nested = root.join("pkg/.ruff_cache");
        fs::create_dir_all(nested.join("0.0.0/check")).unwrap();
        fs::write(nested.join("CACHEDIR.TAG"), "").unwrap();
        let untagged = root.join("pkg/.ruff_cache_not");
        fs::create_dir_all(&untagged).unwrap();
        let mut settings = Configuration::default().into_settings(&root).unwrap();
        settings.cache_dir = root.join("cache");
        // Relative to the current directory, as `--cache-dir` may be.
        #[cfg(unix)]
        {
            let cwd = std::env::current_dir().unwrap();
            let up: PathBuf = cwd.components().skip(1).map(|_| "..").collect();
            settings.cache_dir = up.join(settings.cache_dir.strip_prefix("/").unwrap());
        }
        let root_settings = RootSettings {
            strategy: PyprojectDiscoveryStrategy::Fixed,
            resolved: ResolvedSettings::new(settings),
        };
        fs::create_dir_all(root.join("cache/0.0.0")).unwrap();
        let later = Timestamp::now() + Duration::from_secs(10);
        let recorder = WalkRecorder::new(later, std::slice::from_ref(&root), &root_settings);
        for dir in [
            root.join("cache"),
            root.join("cache/0.0.0"),
            nested.clone(),
            nested.join("0.0.0/check"),
            untagged.clone(),
        ] {
            recorder.listing(&dir);
        }
        let inputs = recorder.finish().unwrap();
        let recorded: Vec<&Path> = inputs.paths().collect();
        assert!(recorded.contains(&untagged.as_path()), "{recorded:#?}");
        assert!(
            !recorded
                .iter()
                .any(|path| path.starts_with(root.join("cache")) || path.starts_with(&nested)),
            "{recorded:#?}"
        );
    }

    #[test]
    fn a_worktree_records_its_common_git_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let common = root.join("repo/.git");
        let git_dir = common.join("worktrees/wt");
        fs::create_dir_all(&git_dir).unwrap();
        fs::create_dir_all(common.join("info")).unwrap();
        fs::write(common.join("info/exclude"), "x\n").unwrap();
        fs::write(git_dir.join("commondir"), "../..\n").unwrap();
        let worktree = root.join("wt");
        fs::create_dir_all(&worktree).unwrap();
        fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", git_dir.display()),
        )
        .unwrap();

        let settings = Configuration::default().into_settings(&worktree).unwrap();
        let root_settings = RootSettings {
            strategy: PyprojectDiscoveryStrategy::Fixed,
            resolved: ResolvedSettings::new(settings),
        };
        let later = Timestamp::now() + Duration::from_secs(10);
        let recorder = WalkRecorder::new(later, std::slice::from_ref(&worktree), &root_settings);
        ruff_workspace::pyproject::DiscoveryObserver::listing(&recorder, &worktree);
        let inputs = recorder.finish().unwrap();
        let exclude = git_dir.join("../../info/exclude");
        assert!(inputs.paths().any(|path| path == exclude), "{inputs:#?}");
        fs::write(common.join("info/exclude"), "y\n").unwrap();
        assert!(inputs.recheck(later).is_none());
    }
}
