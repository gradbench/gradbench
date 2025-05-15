use std::{
    fs,
    io::Write,
    process::{Command, ExitCode},
};

use colored::Colorize;
use similar::{ChangeTag, TextDiff};
use tempfile::NamedTempFile;

use crate::err_fail;

// These strings should all be the same length.
const RUNNING: &str = "running";
const PASSED: &str = " passed";
const FAILED: &str = " failed";
const MISSING: &str = "missing";

pub struct Config {
    fix: bool,
    name: Option<&'static str>,
}

impl Config {
    fn name(&mut self, name: &'static str) {
        println!("{} {name}", RUNNING.bold());
        self.name = Some(name);
    }
}

type Lint = fn(&mut Config) -> anyhow::Result<bool>;

pub struct Lints {
    all: Vec<Lint>,
    chosen: Vec<Lint>,
}

impl Lints {
    pub fn new() -> Self {
        Self {
            all: Vec::new(),
            chosen: Vec::new(),
        }
    }

    pub fn flag(&mut self, choose: bool, function: Lint) {
        self.all.push(function);
        if choose {
            self.chosen.push(function);
        }
    }

    pub fn run(self, fix: bool) -> Result<(), ExitCode> {
        let mut lints = self.chosen;
        if lints.is_empty() {
            lints = self.all;
        }
        let mut fails = Vec::new();
        let mut misses = Vec::new();
        let mut cfg = Config { fix, name: None };
        let mut first = true;
        for lint in lints {
            if !first {
                println!();
            }
            first = false;
            let result = lint(&mut cfg);
            let name = cfg.name.take().expect("a linter forgot to say its name");
            match result {
                Ok(true) => println!("{}", format!("{} {name}", PASSED.bold()).green()),
                Ok(false) => {
                    println!("{}", format!("{} {name}", FAILED.bold()).red());
                    fails.push(name);
                }
                Err(error) => {
                    err_fail(error);
                    println!("{}", format!("{} {name}", MISSING.bold()).yellow());
                    misses.push(name);
                }
            }
        }
        if !misses.is_empty() {
            println!();
            println!("{}", format!("{MISSING} lints").bold().yellow());
            for miss in &misses {
                println!("  {}", miss.yellow());
            }
        }
        if !fails.is_empty() {
            println!();
            println!("{}", format!("{FAILED} lints").bold().red());
            for fail in fails {
                println!("  {}", fail.red());
            }
            Err(ExitCode::FAILURE)
        } else if !misses.is_empty() {
            Err(ExitCode::from(2))
        } else {
            Ok(())
        }
    }
}

pub fn clang_format(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("clang-format");
    let files = String::from_utf8(
        Command::new("git")
            .args(["ls-files", "*.c", "*.cpp", "*.h", "*.hpp"])
            .output()?
            .stdout,
    )?;
    let run = |name| {
        let mut cmd = Command::new(name);
        if cfg.fix {
            cmd.arg("-i");
        } else {
            cmd.args(["--dry-run", "-Werror"]);
        }
        cmd.args(files.lines());
        Ok(cmd.status()?.success())
    };
    run("clang-format-19").or_else(|_| run("clang-format"))
}

pub fn clippy(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("Clippy");
    Ok(Command::new("cargo").arg("clippy").status()?.success())
}

pub fn eslint(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("ESLint");
    Ok(Command::new("bun")
        .args(["run", "--filter=@gradbench/website", "lint"])
        .status()?
        .success())
}

pub fn markdown_toc(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("markdown-toc");
    let mut passed = true;
    for filename in ["README.md", "CONTRIBUTING.md"] {
        let mut cmd = Command::new("bun");
        cmd.args(["run", "markdown-toc", "--bullets=-", "-i"]);
        if cfg.fix {
            cmd.arg(filename);
            cmd.status()?;
        } else {
            let before = fs::read_to_string(filename)?;
            let after = {
                let mut tmp = NamedTempFile::new()?;
                tmp.write_all(before.as_bytes())?;
                cmd.arg(tmp.path().as_os_str());
                cmd.status()?;
                fs::read_to_string(tmp)?
            };
            if before != after {
                passed = false;
                println!("{filename}");
                let diff = TextDiff::from_lines(&before, &after);
                for group in diff.grouped_ops(3) {
                    for op in group {
                        for change in diff.iter_changes(&op) {
                            match change.tag() {
                                ChangeTag::Equal => print!(" {}", change.value().dimmed()),
                                ChangeTag::Delete => print!("-{}", change.value().red()),
                                ChangeTag::Insert => print!("+{}", change.value().green()),
                            }
                        }
                    }
                }
                println!();
            }
        }
    }
    Ok(passed)
}

pub fn prettier(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("Prettier");
    let mut cmd = Command::new("bun");
    cmd.args(["run", "prettier", "."]);
    cmd.arg(if cfg.fix { "--write" } else { "--check" });
    Ok(cmd.status()?.success())
}

pub fn ruff_check(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("Ruff linter");
    let mut cmd = Command::new("uv");
    cmd.args(["run", "ruff", "check"]);
    if cfg.fix {
        cmd.arg("--fix");
    }
    Ok(cmd.status()?.success())
}

pub fn ruff_format(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("Ruff formatter");
    let mut cmd = Command::new("uv");
    cmd.args(["run", "ruff", "format"]);
    if !cfg.fix {
        cmd.arg("--check");
    }
    Ok(cmd.status()?.success())
}

pub fn rustfmt(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("Rustfmt");
    let mut cmd = Command::new("cargo");
    cmd.arg("fmt");
    if !cfg.fix {
        cmd.arg("--check");
    }
    Ok(cmd.status()?.success())
}

pub fn typescript(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("TypeScript");
    Ok(Command::new("bun")
        .args(["run", "--filter=*", "typecheck"])
        .status()?
        .success())
}
