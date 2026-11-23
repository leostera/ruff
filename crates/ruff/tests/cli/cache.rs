//! Runs with the cache print, exit and write what the same runs without it do, for each kind of
//! edit a developer makes between runs: the edit is made to two copies of a project, and every
//! command runs in one with the cache and in the other without it.
//!
//! Each command runs on the project before the edit, so that the cache knows it, then three times
//! after: twice inside the racy window, where a `stat` can't be trusted, and once past it.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, SystemTime};

use filetime::FileTime;
use insta_cmd::get_cargo_bin;
use test_case::test_case;

use super::BIN_NAME;

/// The project every scenario starts from: clean and formatted, apart from `pkg/dirty.py`.
const PROJECT: &[(&str, &str)] = &[
    (
        "pyproject.toml",
        "[tool.ruff]\nextend = \"base.toml\"\nsrc = [\"src\"]\n\n[tool.ruff.lint]\nextend-select = [\"I\"]\n",
    ),
    ("base.toml", "line-length = 88\n"),
    ("pkg/__init__.py", ""),
    (
        "pkg/clean.py",
        "import bar\nimport foo\n\nprint(bar, foo)\n",
    ),
    ("pkg/other.py", "def f(x):\n    return x + 1\n"),
    ("pkg/dirty.py", "import os\n"),
    ("pkg/sub/__init__.py", ""),
    ("pkg/sub/deep.py", "VALUE = [1, 2, 3]\n"),
    ("src/.keep", ""),
    ("scripts/run.py", "print(\"hello\")\n"),
];

/// What a command finds: what it prints, how it exits, and the project it leaves.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    status: Option<i32>,
    stdout: String,
    stderr: String,
    tree: BTreeMap<String, String>,
}

/// A copy of [`PROJECT`].
struct Project {
    root: PathBuf,
    /// The cache directory, or `None` to run without the cache.
    cache: Option<PathBuf>,
}

impl Project {
    /// Creates the project in `root`, its files a minute old, as a developer's are.
    fn new(root: PathBuf, cache: Option<PathBuf>) -> Self {
        let project = Self { root, cache };
        project.reset(&Self::initial());
        let past = FileTime::from_system_time(SystemTime::now() - Duration::from_secs(60));
        for (path, _) in PROJECT {
            filetime::set_file_times(project.root.join(path), past, past).unwrap();
        }
        fs::create_dir_all(project.root.join(".git/info")).unwrap();
        project
    }

    /// Returns [`PROJECT`]'s files.
    fn initial() -> BTreeMap<String, String> {
        PROJECT
            .iter()
            .map(|(path, content)| ((*path).to_string(), (*content).to_string()))
            .collect()
    }

    /// Returns `path` in the project.
    fn at(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }

    /// Writes `content` to `path`, creating its directory.
    fn write(&self, path: &str, content: &str) {
        let path = self.at(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    /// Puts back the files of `tree` a command changed, leaving the others untouched.
    fn reset(&self, tree: &BTreeMap<String, String>) {
        let now = self.tree();
        for (path, content) in tree {
            if now.get(path) != Some(content) {
                self.write(path, content);
            }
        }
        for path in now.keys().filter(|path| !tree.contains_key(*path)) {
            fs::remove_file(self.at(path)).unwrap();
        }
    }

    /// Returns the project's files and their contents, apart from Git's and the cache's.
    fn tree(&self) -> BTreeMap<String, String> {
        let mut tree = BTreeMap::new();
        let mut dirs = vec![self.root.clone()];
        while let Some(dir) = dirs.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                let relative = path.strip_prefix(&self.root).unwrap();
                let relative = relative.to_str().unwrap().replace('\\', "/");
                if relative == ".git" || relative == ".ruff_cache" {
                    continue;
                }
                if path.is_dir() {
                    dirs.push(path);
                } else {
                    tree.insert(relative, fs::read_to_string(&path).unwrap_or_default());
                }
            }
        }
        tree
    }

    /// Runs Ruff with `args`, and puts back what it wrote.
    fn run(&self, args: &[&str]) -> Outcome {
        let before = self.tree();
        let mut command = Command::new(get_cargo_bin(BIN_NAME));
        command.current_dir(&self.root).env_clear().args(args);
        match &self.cache {
            Some(cache) => command.arg("--cache-dir").arg(cache),
            None => command.arg("--no-cache"),
        };
        let output = command.output().unwrap();
        let mask = |bytes: &[u8]| {
            String::from_utf8_lossy(bytes).replace(self.root.to_str().unwrap(), "[PROJECT]")
        };
        let outcome = Outcome {
            status: output.status.code(),
            stdout: mask(&output.stdout),
            stderr: mask(&output.stderr),
            tree: self.tree(),
        };
        self.reset(&before);
        outcome
    }
}

/// The commands each scenario runs.
const COMMANDS: &[&[&str]] = &[
    &["check"],
    &["check", "pkg"],
    &["check", "pkg/clean.py", "pkg/other.py"],
    &["check", "--fix"],
    &["check", "--statistics"],
    &["format", "--check"],
    &["format", "--check", "pkg"],
    &["format"],
];

/// How long a run waits for its `stat`s to be trusted.
const RACY_WINDOW: Duration = Duration::from_millis(2_100);

/// Runs every command in `cached` and `uncached` and asserts they agree.
fn round(edit: &str, label: &str, cached: &Project, uncached: &Project) {
    for args in COMMANDS {
        let expected = uncached.run(args);
        let actual = cached.run(args);
        assert_eq!(
            actual,
            expected,
            "{edit} / ruff {} / {label}: the cached run differs",
            args.join(" ")
        );
    }
}

#[test_case("nothing", |_| {}; "nothing")]
#[test_case("a clean new file", |p| p.write("pkg/new.py", "X = 1\n"); "new_clean")]
#[test_case("a new file with a diagnostic", |p| p.write("pkg/new.py", "import sys\n"); "new_dirty")]
#[test_case("a syntax error", |p| p.write("pkg/other.py", "def f(:\n"); "syntax_error")]
#[test_case("a clean file given a diagnostic", |p| p.write("pkg/other.py", "import sys\ndef f(x):\n    return x + 1\n"); "clean_to_dirty")]
#[test_case("a clean file left clean", |p| p.write("pkg/other.py", "def f(x):\n    return x + 2\n"); "clean_to_clean")]
#[test_case("a clean file left unformatted", |p| p.write("pkg/other.py", "def f(x):\n    return x+1\n"); "unformatted")]
#[test_case("a dirty file fixed", |p| p.write("pkg/dirty.py", ""); "dirty_to_clean")]
#[test_case("a same-size edit with the mtime put back", same_size_edit; "same_size")]
#[test_case("a file renamed", |p| fs::rename(p.at("pkg/other.py"), p.at("pkg/renamed.py")).unwrap(); "renamed")]
#[test_case("a clean file deleted", |p| fs::remove_file(p.at("pkg/other.py")).unwrap(); "deleted")]
#[test_case("a directory deleted", |p| fs::remove_dir_all(p.at("pkg/sub")).unwrap(); "dir_deleted")]
#[test_case("an ignored file with a diagnostic", |p| {
    p.write(".gitignore", "pkg/ignored.py\n");
    p.write("pkg/ignored.py", "import sys\n");
}; "ignored")]
#[test_case("a nested .gitignore ignoring a dirty file", |p| p.write("pkg/.gitignore", "dirty.py\n"); "nested_gitignore")]
#[test_case("Git's info/exclude ignoring a dirty file", |p| p.write(".git/info/exclude", "dirty.py\n"); "info_exclude")]
#[test_case("the configuration changed", |p| p.write("pyproject.toml", "[tool.ruff]\nextend = \"base.toml\"\nsrc = [\"src\"]\n\n[tool.ruff.lint]\nextend-select = [\"I\", \"T20\"]\n"); "config")]
#[test_case("an extended configuration changed", |p| p.write("base.toml", "line-length = 10\n"); "extended_config")]
#[test_case("a nested configuration added", |p| p.write("pkg/sub/ruff.toml", "[lint]\nselect = [\"C4\"]\n[format]\nquote-style = \"single\"\n"); "nested_config")]
#[test_case("an exclusion added", |p| p.write("base.toml", "line-length = 88\nextend-exclude = [\"pkg/dirty.py\"]\n"); "exclude")]
#[test_case("a first-party module appears", |p| fs::create_dir_all(p.at("src/foo")).unwrap(); "first_party_dir")]
#[test_case("a first-party module file appears", |p| p.write("src/foo.py", ""); "first_party_file")]
#[test_case("a package root appears", |p| p.write("scripts/__init__.py", ""); "package_root")]
fn a_cached_run_matches_an_uncached_run(edit: &str, apply: fn(&Project)) {
    let base = tempfile::tempdir().unwrap();
    let base = dunce::canonicalize(base.path()).unwrap();
    let cached = Project::new(base.join("cached"), Some(base.join("cache")));
    let uncached = Project::new(base.join("uncached"), None);

    // A project older than the racy window, known to the cache by every command.
    std::thread::sleep(RACY_WINDOW);
    round(edit, "before the edit", &cached, &uncached);

    apply(&cached);
    apply(&uncached);
    assert_eq!(cached.tree(), uncached.tree());
    round(edit, "first run after the edit", &cached, &uncached);
    round(edit, "second run after the edit", &cached, &uncached);
    std::thread::sleep(RACY_WINDOW);
    round(edit, "past the racy window", &cached, &uncached);
}

/// Rewrites `pkg/other.py` with a diagnostic, the same size, and puts its modification time back:
/// only its `ctime` tells.
fn same_size_edit(project: &Project) {
    let path = project.at("pkg/other.py");
    let mtime = FileTime::from_last_modification_time(&fs::metadata(&path).unwrap());
    let before = fs::read_to_string(&path).unwrap();
    let after = "import a\ndef f(x):return x\n";
    assert_eq!(before.len(), after.len());
    fs::write(&path, after).unwrap();
    filetime::set_file_mtime(&path, mtime).unwrap();
}

/// Makes `scripts/run.py` executable without a shebang, which `EXE002` reports.
#[cfg(unix)]
#[test]
fn a_file_made_executable_is_checked_again() {
    use std::os::unix::fs::PermissionsExt;

    let base = tempfile::tempdir().unwrap();
    let base = dunce::canonicalize(base.path()).unwrap();
    let cached = Project::new(base.join("cached"), Some(base.join("cache")));
    let uncached = Project::new(base.join("uncached"), None);
    let args: &[&str] = &["check", "--extend-select", "EXE"];
    std::thread::sleep(RACY_WINDOW);
    assert_eq!(cached.run(args), uncached.run(args));
    for project in [&cached, &uncached] {
        let path = project.at("scripts/run.py");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert_eq!(cached.run(args), uncached.run(args));
    std::thread::sleep(RACY_WINDOW);
    assert_eq!(cached.run(args), uncached.run(args));
}

/// Runs that differ in a command-line override that leaves the root settings as they are but
/// changes a nested configuration's: they share a run shape, and neither can reuse the other's
/// walk.
#[test_case(&["--exclude", "deep.py"]; "exclude")]
#[test_case(&["--extend-select", "I"]; "select")]
fn an_override_that_only_changes_a_nested_configuration(overrides: &[&str]) {
    let base = tempfile::tempdir().unwrap();
    let base = dunce::canonicalize(base.path()).unwrap();
    let cached = Project::new(base.join("cached"), Some(base.join("cache")));
    let uncached = Project::new(base.join("uncached"), None);
    for project in [&cached, &uncached] {
        project.write("base.toml", "line-length = 88\nexclude = [\"deep.py\"]\n");
        project.write("pkg/sub/ruff.toml", "[lint]\nselect = [\"F\"]\n");
        // Unsorted imports, which only `I` reports, and an unused one.
        project.write("pkg/sub/deep.py", "import sys\nimport os\n\nprint(sys)\n");
    }
    let plain: &[&str] = &["check"];
    let overridden: Vec<&str> = plain.iter().chain(overrides).copied().collect();
    for args in [plain, &overridden] {
        cached.run(args);
    }
    std::thread::sleep(RACY_WINDOW);
    for args in [plain, &overridden, plain, &overridden] {
        assert_eq!(
            cached.run(args),
            uncached.run(args),
            "ruff {}",
            args.join(" ")
        );
    }
}
