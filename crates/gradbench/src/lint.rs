use std::{
    ffi::OsStr,
    fs,
    io::Write,
    process::{Command, ExitCode},
};

use anyhow::anyhow;
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

fn bun(cmd: &mut Command) -> anyhow::Result<bool> {
    Ok(cmd
        .status()
        .map_err(|_| {
            if Command::new("bun").arg("--version").output().is_ok() {
                anyhow!("you must run `bun install`")
            } else {
                anyhow!("install Bun from https://bun.sh/ and then run `bun install`")
            }
        })?
        .success())
}

fn node_bin(name: &str, f: impl FnOnce(&mut Command)) -> anyhow::Result<bool> {
    let mut cmd = Command::new(format!("node_modules/.bin/{name}"));
    f(&mut cmd);
    bun(&mut cmd)
}

fn uv(f: impl Fn(&mut Command)) -> anyhow::Result<bool> {
    let run = |cmd: &mut Command| {
        cmd.arg("run");
        f(cmd);
        Ok(cmd
            .status()
            .map_err(|_| anyhow!("install uv from https://docs.astral.sh/uv/"))?
            .success())
    };
    run(Command::new("steam-run").arg("uv")).or_else(|_| run(&mut Command::new("uv")))
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
    bun(Command::new("bun").args(["run", "--filter=@gradbench/website", "lint"]))
}

pub fn markdown_toc(cfg: &mut Config) -> anyhow::Result<bool> {
    fn run(filename: impl AsRef<OsStr>) -> anyhow::Result<()> {
        if node_bin("markdown-toc", |cmd| {
            cmd.args(["--bullets=-", "-i"]);
            cmd.arg(filename);
        })? {
            Ok(())
        } else {
            Err(anyhow!("markdown-toc failed"))
        }
    }

    cfg.name("markdown-toc");
    let mut passed = true;
    for filename in ["README.md", "CONTRIBUTING.md"] {
        if cfg.fix {
            run(filename)?;
        } else {
            let before = fs::read_to_string(filename)?;
            let after = {
                let mut tmp = NamedTempFile::new()?;
                tmp.write_all(before.as_bytes())?;
                run(tmp.path().as_os_str())?;
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
    node_bin("prettier", |cmd| {
        cmd.arg(".");
        cmd.arg(if cfg.fix { "--write" } else { "--check" });
    })
}

pub fn ruff_check(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("Ruff linter");
    uv(|cmd| {
        cmd.args(["ruff", "check"]);
        if cfg.fix {
            cmd.arg("--fix");
        }
    })
}

pub fn ruff_format(cfg: &mut Config) -> anyhow::Result<bool> {
    cfg.name("Ruff formatter");
    uv(|cmd| {
        cmd.args(["ruff", "format"]);
        if !cfg.fix {
            cmd.arg("--check");
        }
    })
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
    bun(Command::new("bun").args(["run", "--filter=*", "typecheck"]))
}
