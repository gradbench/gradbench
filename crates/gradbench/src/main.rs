mod intermediary;
mod lint;
mod log;
mod protocol;
mod stats;
mod util;

use std::{
    backtrace::BacktraceStatus,
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    env,
    fmt::Write as _,
    fs,
    io,
    path::{Path, PathBuf},
    process::{Command, ExitCode, ExitStatus, Output, Stdio},
    rc::Rc,
    str::FromStr,
    time::Duration,
};

use anyhow::{anyhow, bail, Context};
use clap::{Parser, Subcommand};
use colored::Colorize;
use itertools::Itertools;
use serde::Serialize;
use stats::StatsMetadata;
use strum::{EnumIter, EnumString, IntoStaticStr};
use util::{run_in_out, shlex_cmd, CtrlC};

/// CLI utilities for GradBench, a benchmark suite for differentiable programming across languages
/// and domains.
///
/// When working in a clone of the GradBench repository, replace `gradbench` with `./gradbench` in
/// usage documentation.
#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Help text for the `outcome` argument of the `exit-code` subcommand.
const OUTCOME_HELP: &str =
    "One of `interrupt`, `timeout`, `invalid`, `failure`, `undefined`, `error`, or `success`";

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run an eval locally using the Nix-provided environment.
    Eval {
        /// The name of the eval to run
        eval: String,

        /// Arguments for the eval itself
        args: Vec<String>,
    },

    /// Run a tool locally using the Nix-provided environment.
    Tool {
        /// The name of the tool to run
        tool: String,

        /// Arguments for the tool itself
        args: Vec<String>,
    },

    /// Run a given tool on a given eval. For example:
    ///
    ///     gradbench run -o log.jsonl --eval "gradbench eval hello" --tool "gradbench tool pytorch"
    #[clap(about = "Run a given tool on a given eval", verbatim_doc_comment)]
    Run {
        /// A shell script to run the eval. For example: `gradbench eval hello`
        #[clap(long, verbatim_doc_comment)]
        eval: String,

        /// A shell script to run the tool. For example: `gradbench tool pytorch`
        #[clap(long)]
        tool: String,

        /// A path to save the full log. For example: `log.jsonl`
        #[clap(short, long)]
        output: Option<PathBuf>,

        /// The timeout, in seconds, for tool responses (not implemented on Windows)
        #[clap(long)]
        timeout: Option<u64>,
    },

    /// Return a `gradbench run` exit code corresponding to a specific outcome.
    ExitCode {
        #[clap(help = OUTCOME_HELP)]
        outcome: String,
    },

    /// Perform useful operations on the log files produced by `gradbench run`.
    Log {
        #[command(subcommand)]
        command: LogCommands,
    },

    /// Perform a task in a clone of the https://github.com/gradbench/gradbench repository.
    ///
    /// These subcommands will first attempt to check that the current working directory is the root
    /// of a clone of the GradBench repository, exiting with an error if not.
    Repo {
        #[command(subcommand)]
        command: RepoCommands,
    },
}

#[derive(Debug, Subcommand)]
enum RepoCommands {
    /// Build and run one or more evals against one of more tools, locally.
    ///
    /// The `--eval` and `--tool` arguments can each be repeated any number of times, and each
    /// instance can take any of the following forms:
    ///
    /// - A name, e.g. `--eval foo`.
    ///
    /// - A name followed by some arguments, e.g. `--tool "foo --bar --baz=qux"`.
    ///
    /// - A dollar sign followed by any command, e.g. `--eval "$ echo 'an example'"`.
    ///
    /// The output directory will contain a `<EVAL>/<TOOL>.jsonl` file for each `<EVAL>` and each
    /// `<TOOL>`, where the eval and tool names are mangled to only contain ASCII letters, digits,
    /// and hyphens. It is an error for two mangled names to conflict.
    Run {
        /// One or more evals to run, or all evals by default
        #[clap(short, long)]
        eval: Vec<String>,

        /// One or more tools to run, or all tools by default
        #[clap(short, long)]
        tool: Vec<String>,

        /// Evals to omit
        #[clap(long, value_name = "EVAL")]
        no_eval: Vec<String>,

        /// Tools to omit
        #[clap(long, value_name = "TOOL")]
        no_tool: Vec<String>,

        /// Output directory
        #[clap(short, long)]
        output: Option<PathBuf>,

        /// The timeout, in seconds, for tool responses (not implemented on Windows)
        #[clap(long)]
        timeout: Option<u64>,

        /// Only allow known named evals and tools, and check against their expected outcome
        #[clap(long)]
        check: bool,

        /// Print commands to stdout instead of running anything
        #[clap(long)]
        dry_run: bool,
    },

    /// Build (if needed) and run an eval locally.
    Eval {
        /// The name of the eval to run
        eval: String,

        /// Arguments for the eval itself
        args: Vec<String>,
    },

    /// Build (if needed) and run a tool locally.
    Tool {
        /// The name of the tool to run
        tool: String,

        /// Arguments for the tool itself
        args: Vec<String>,
    },

    /// Prepare an eval for local use (usually a no-op).
    BuildEval {
        /// The name of the eval to build
        eval: String,
    },

    /// Prepare a tool for local use (may build dependencies).
    BuildTool {
        /// The name of the tool to build
        tool: String,
    },

    /// Run linters on the codebase.
    ///
    /// By default, every linter is run and no changes are made. Use the `--fix` flag to autofix
    /// when possible. Use other flags to only run specific linters. In any case, the exit code is 0
    /// if everything passed, 1 if any lints failed, or 2 if no lints failed but not all linters
    /// could be run successfully.
    Lint {
        /// Automatically fix everything possible
        #[clap(long)]
        fix: bool,

        /// Run only clang-format
        #[clap(long)]
        clang_format: bool,

        /// Run only Clippy
        #[clap(long)]
        clippy: bool,

        /// Run only ESLint
        #[clap(long)]
        eslint: bool,

        /// Run only markdown-toc
        #[clap(long)]
        markdown_toc: bool,

        /// Run only nixfmt
        #[clap(long)]
        nixfmt: bool,

        /// Run only Prettier
        #[clap(long)]
        prettier: bool,

        /// Run only the Ruff linter
        #[clap(long)]
        ruff_check: bool,

        /// Run only the Ruff formatter
        #[clap(long)]
        ruff_format: bool,

        /// Run only runic
        #[clap(long)]
        runic: bool,

        /// Run only Rustfmt
        #[clap(long)]
        rustfmt: bool,

        /// Run only TypeScript
        #[clap(long)]
        typescript: bool,
    },

    /// Print JSON values for consumption in GitHub Actions.
    ///
    /// Each value is printed on a single line, preceded by the name of that value and an equals
    /// sign. No extra whitespace is printed, because GitHub Actions seems to be sensitive to that.
    Matrix,

    /// Generate summary data files and plots from a directory containing log files.
    ///
    /// The directory should contain a `<EVAL>/<TOOL>.jsonl` file for each `<EVAL>` under `evals`
    /// and each `<TOOL>` under `tools`.
    Stats {
        /// The directory containing log files
        input: PathBuf,

        /// The directory to create
        #[clap(short, long)]
        output: PathBuf,

        /// The current date
        #[clap(long)]
        date: Option<String>,

        /// The source Git commit SHA
        #[clap(long)]
        commit: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum LogCommands {
    /// Remove input/output fields from "evaluate" messages and responses.
    ///
    /// Writes to stdout unless the `--output` option is used. It is expected that the input log
    /// file is well-formed, but not that it corresponds to a successful run. In particular, the
    /// final message may not have a response - this occurs when the tool crashes or times out
    /// before it gets to respond.
    Trim {
        /// The input log file
        input: Option<PathBuf>,

        /// The output log file
        #[clap(short, long)]
        output: Option<PathBuf>,
    },

    /// Print a human-readable summary of the log file, including the eval, tool, configuration,
    /// etc.
    ///
    /// Will fail with a not necessarily very friendly error if the log file is malformed.
    Summary {
        /// The input log file
        input: Option<PathBuf>,
    },

    /// Move log files from a directory with three layers of nesting to a directory with only two.
    Flatten {
        /// The input directory
        input: PathBuf,

        /// The output directory
        #[clap(short, long)]
        output: PathBuf,
    },
}

/// Print `error` to stderr, then return [`ExitCode::FAILURE`].
fn err_fail(error: anyhow::Error) -> ExitCode {
    eprintln!("{error:#}");
    let backtrace = error.backtrace();
    match backtrace.status() {
        BacktraceStatus::Disabled => eprintln!(
            "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace"
        ),
        BacktraceStatus::Captured => eprint!("{backtrace}"),
        _ => {}
    }
    ExitCode::FAILURE
}

/// Return `Err` if the status is not successful, preserving its exit code whenever possible.
fn status_code(status: ExitStatus) -> Result<(), ExitCode> {
    if status.success() {
        Ok(())
    } else {
        match status.code() {
            Some(code) => match u8::try_from(code) {
                Ok(value) => Err(value.into()),
                Err(_) => Err(ExitCode::FAILURE),
            },
            None => Err(ExitCode::FAILURE),
        }
    }
}

/// Run a command and preserve its exit code whenever possible.
fn run(command: &mut Command) -> Result<Output, ExitCode> {
    let output = command
        .spawn()
        .and_then(|child| child.wait_with_output())
        .with_context(|| format!("error running {:?}", command.get_program()))
        .map_err(err_fail)?;
    status_code(output.status)?;
    Ok(output)
}

/// Return the path to the Nix file for an eval.
fn nix_eval_path(eval: &str) -> PathBuf {
    Path::new("evals").join(eval).join("default.nix")
}

/// Return the path to the Nix file for a tool.
fn nix_tool_path(tool: &str) -> PathBuf {
    Path::new("tools").join(tool).join("default.nix")
}

/// Return the flake attribute name for an eval.
fn nix_eval_attr(eval: &str) -> String {
    format!("eval-{eval}")
}

/// Return the flake attribute name for a tool.
fn nix_tool_attr(tool: &str) -> String {
    format!("tool-{tool}")
}

/// Create a `nix build` command for a flake attribute.
fn nix_build_cmd(attr: &str, offline: bool) -> Command {
    let mut cmd = Command::new("nix");
    cmd.args(["build", "--no-link", "--print-out-paths"]);
    if offline {
        cmd.arg("--offline");
    }
    cmd.arg(format!(".#{attr}"));
    cmd
}

/// Build a Nix derivation and return its output path.
fn nix_build(attr: &str, offline: bool) -> Result<PathBuf, ExitCode> {
    let mut cmd = nix_build_cmd(attr, offline);
    cmd.stdout(Stdio::piped());
    let output = run(&mut cmd)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let out = stdout
        .lines()
        .next()
        .ok_or_else(|| err_fail(anyhow!("nix build produced no output for {attr:?}")))?;
    Ok(PathBuf::from(out))
}

/// Build a Nix eval derivation and return its output path.
fn build_eval(eval: &str, offline: bool) -> Result<PathBuf, ExitCode> {
    let path = nix_eval_path(eval);
    if !path.exists() {
        return Err(err_fail(anyhow!("can't find eval Nix file: {path:?}")));
    }
    nix_build(&nix_eval_attr(eval), offline)
}

/// Build a Nix tool derivation and return its output path.
fn build_tool(tool: &str, offline: bool) -> Result<PathBuf, ExitCode> {
    let path = nix_tool_path(tool);
    if !path.exists() {
        return Err(err_fail(anyhow!("can't find tool Nix file: {path:?}")));
    }
    nix_build(&nix_tool_attr(tool), offline)
}

/// Return a run command for a built Nix derivation.
fn run_cmd(output: &Path, args: &[String]) -> Command {
    let mut cmd = Command::new(output.join("bin/run"));
    cmd.args(args);
    cmd
}

/// An imperfect outcome from running the intermediary.
#[derive(Clone, Copy, Debug, EnumIter, EnumString, Eq, IntoStaticStr, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
enum BadOutcome {
    /// The user sent an interrupt signal.
    Interrupt,

    /// The tool timed out.
    Timeout,

    /// The tool returned some number of invalid results for the eval.
    Invalid,

    /// The tool failed to evaluate a function.
    Failure,

    /// The tool failed to define a module.
    Undefined,

    /// Some other error occurred. Any relevant information has already been printed.
    Error,
}

impl From<BadOutcome> for ExitCode {
    fn from(outcome: BadOutcome) -> Self {
        match outcome {
            BadOutcome::Interrupt => ExitCode::from(6),
            BadOutcome::Timeout => ExitCode::from(5),
            BadOutcome::Invalid => ExitCode::from(4),
            BadOutcome::Failure => ExitCode::from(3),
            BadOutcome::Undefined => ExitCode::from(2),
            BadOutcome::Error => ExitCode::from(1),
        }
    }
}

/// Check that the current working directory is the root of a Git repository.
fn check_git() -> Result<(), ExitCode> {
    if Path::new(".git").exists() {
        Ok(())
    } else {
        eprintln!("error running a repo subcommand: current working directory is not a Git repository root");
        Err(ExitCode::FAILURE)
    }
}

/// Return a command that runs its argument as a shell command.
fn shell(command: &str) -> anyhow::Result<Command> {
    match shlex::split(command).map(VecDeque::from) {
        Some(mut parts) => match parts.pop_front() {
            Some(first) => {
                let mut cmd = Command::new(first);
                cmd.args(parts);
                Ok(cmd)
            }
            None => Err(anyhow!("empty command")),
        },
        None => Err(anyhow!("failed to split command")),
    }
}

/// Convert command parts into a `Command`.
fn command_from_parts(parts: &[String]) -> anyhow::Result<Command> {
    let (first, rest) = parts
        .split_first()
        .ok_or_else(|| anyhow!("empty command parts"))?;
    let mut cmd = Command::new(first);
    cmd.args(rest);
    Ok(cmd)
}

/// Configure a subcommand to be run by the intermediary.
fn configure_intermediary_subcommand(cmd: &mut Command) {
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(cmd, 0);
}

/// List the entries in a directory.
fn ls(dir: &str) -> anyhow::Result<Vec<String>> {
    fs::read_dir(dir)
        .with_context(|| format!("error reading directory {dir:?}"))?
        .map(|entry| {
            entry?
                .file_name()
                .into_string()
                .map_err(|name| anyhow!("invalid file name {name:?}"))
        })
        .collect()
}

/// Config for running one or more evals against one or more tools.
#[derive(Default)]
struct RunConfig {
    /// Output directory.
    output: Option<PathBuf>,

    /// The timeout, in seconds, for tool responses (not implemented on Windows).
    timeout: Option<u64>,

    /// Only allow known named evals and tools, and check against their expected outcome.
    check: bool,
}

/// Raw lists of evals and tools to run against each other.
struct RunRaw {
    /// Evals to run, or all by default.
    eval: Vec<String>,

    /// Tools to run, or all by default.
    tool: Vec<String>,

    /// Evals to omit.
    no_eval: Vec<String>,

    /// Tools to omit.
    no_tool: Vec<String>,

    /// Don't actually run anything, just print commands to stdout.
    dry_run: bool,
}

/// Choice between talking about evals or talking about tools.
enum RunItemKind {
    /// Evals.
    Eval,

    /// Tools.
    Tool,
}

/// A run command built from a raw `String`.
enum RunCommand {
    /// A named eval or tool that should be run via Nix.
    Named { name: String, args: Vec<String> },

    /// An explicit command provided by the user.
    Command(Vec<String>),
}

/// A raw `String` and the processed run command representing its semantics.
type RunItem = (String, RunCommand);

/// Return a string like the input but with a restricted alphabet.
///
/// The returned string consists only of ASCII letters, digits, and hyphens, and does not start or
/// end with a hyphen. All ASCII alphanumeric characters included in the string; any sequence of
/// other characters is converted to a single hyphen.
fn mangle(string: &str) -> String {
    let mut mangled = String::new();
    let mut pending = false;
    for c in string.chars() {
        if c.is_ascii_alphanumeric() {
            if pending {
                mangled.push('-');
                pending = false;
            }
            mangled.push(c);
        } else if !mangled.is_empty() {
            pending = true;
        }
    }
    mangled
}

/// Given a log directory and a raw eval command string, return a path for that eval's logs.
fn eval_subpath(dir: &Path, eval: &str) -> PathBuf {
    dir.join(mangle(eval))
}

/// Given a log directory and raw eval/tool command strings, return a path for that log.
fn log_subpath(dir: &Path, eval: &str, tool: &str) -> PathBuf {
    let mut path = eval_subpath(dir, eval).join(mangle(tool));
    path.set_extension("jsonl");
    path
}

/// Process a human-friendly list of evals or tools into a deduplicated build list and a run list.
fn process_run_items(
    item_kind: RunItemKind,
    items: Vec<String>,
    omit: Vec<String>,
    default: impl FnOnce() -> anyhow::Result<Vec<String>>,
) -> anyhow::Result<(Vec<String>, Vec<RunItem>)> {
    let kind = match item_kind {
        RunItemKind::Eval => "eval",
        RunItemKind::Tool => "tool",
    };
    let strings = if items.is_empty() {
        let mut all = default()?;
        let set: HashSet<String> = omit.into_iter().collect();
        all.retain(|e| !set.contains(e));
        all.sort();
        all
    } else {
        if !omit.is_empty() {
            bail!("`--no-{kind}` cannot be used together with `--{kind}`");
        }
        items
    };
    let mut mangled = HashSet::new();
    let (builds, runs) = strings
        .into_iter()
        .map(|string| {
            if !mangled.insert(mangle(&string)) {
                let mang = mangle(&string);
                bail!("another `--{kind}` got the same mangled name {mang}: {string:?}");
            }
            let mut parts = VecDeque::from(
                shlex::split(&string)
                    .ok_or_else(|| anyhow!("failed to split `--{kind}`: {string:?}"))?,
            );
            let first = parts
                .pop_front()
                .ok_or_else(|| anyhow!("empty `--{kind}` after splitting: {string:?}"))?;
            let (build, cmd) = if first == "$" {
                let program = parts
                    .pop_front()
                    .ok_or_else(|| anyhow!("empty `--{kind}` after `$`: {string:?}"))?;
                let mut cmd = Vec::new();
                cmd.push(program);
                for arg in parts {
                    cmd.push(arg);
                }
                (None, RunCommand::Command(cmd))
            } else {
                let args = parts.into_iter().collect::<Vec<_>>();
                (
                    Some(first.clone()),
                    RunCommand::Named {
                        name: first,
                        args,
                    },
                )
            };
            Ok((build, (string, cmd)))
        })
        .process_results::<_, _, anyhow::Error, (BTreeSet<Option<String>>, Vec<RunItem>)>(|it| {
            it.unzip()
        })?;
    Ok((builds.into_iter().flatten().collect(), runs))
}

/// Eval and tool outputs from [`process_run_items`].
struct RunItems<'a> {
    /// Evals to build.
    evals_build: &'a [String],

    /// Tools to build.
    tools_build: &'a [String],

    /// Evals to run.
    evals_run: &'a [RunItem],

    /// Tools to run.
    tools_run: &'a [RunItem],
}

/// Print the commands for building and running one more evals against one or more tools.
fn run_dry(
    stdout: &mut impl io::Write,
    this: &str,
    cfg: RunConfig,
    RunItems {
        evals_build,
        tools_build,
        evals_run,
        tools_run,
    }: RunItems,
) -> anyhow::Result<()> {
    for eval in evals_build {
        let cmd = shlex_cmd(&nix_build_cmd(&nix_eval_attr(eval), false))?;
        writeln!(stdout, "{cmd}")?;
    }
    for tool in tools_build {
        let cmd = shlex_cmd(&nix_build_cmd(&nix_tool_attr(tool), false))?;
        writeln!(stdout, "{cmd}")?;
    }
    if let Some(dir) = &cfg.output {
        write!(stdout, "mkdir -p")?;
        for (eval_string, _) in evals_run {
            let subdir = eval_subpath(dir, eval_string);
            let subdir_str = subdir.to_str().ok_or_else(|| {
                anyhow!("failed to convert output directory path to a string: {subdir:?}")
            })?;
            write!(stdout, " {}", shlex::try_quote(subdir_str)?)?;
        }
        writeln!(stdout)?;
    }
    for (eval_string, eval_cmd) in evals_run {
        for (tool_string, tool_cmd) in tools_run {
            let eval = match eval_cmd {
                RunCommand::Named { name, args } => {
                    let build_cmd = shlex_cmd(&nix_build_cmd(&nix_eval_attr(name), true))?;
                    let mut snippet = format!("{build_cmd} | xargs -I{{}} {{}}/bin/run");
                    for arg in args {
                        snippet.push(' ');
                        snippet.push_str(&shlex::try_quote(arg)?);
                    }
                    snippet
                }
                RunCommand::Command(cmd) => shlex::try_join(cmd.iter().map(|s| s.as_str()))?,
            };
            let tool = match tool_cmd {
                RunCommand::Named { name, args } => {
                    let build_cmd = shlex_cmd(&nix_build_cmd(&nix_tool_attr(name), true))?;
                    let mut snippet = format!("{build_cmd} | xargs -I{{}} {{}}/bin/run");
                    for arg in args {
                        snippet.push(' ');
                        snippet.push_str(&shlex::try_quote(arg)?);
                    }
                    snippet
                }
                RunCommand::Command(cmd) => shlex::try_join(cmd.iter().map(|s| s.as_str()))?,
            };
            write!(stdout, "{this} run")?;
            if let Some(seconds) = cfg.timeout {
                write!(stdout, " --timeout {seconds}")?;
            }
            write!(stdout, " --eval {}", shlex::try_quote(&eval)?)?;
            write!(stdout, " --tool {}", shlex::try_quote(&tool)?)?;
            if let Some(dir) = &cfg.output {
                let path = log_subpath(dir, eval_string, tool_string);
                let path_str = path.to_str().ok_or_else(|| {
                    anyhow!("failed to convert output file path to a string: {path:?}")
                })?;
                write!(stdout, " -o {}", shlex::try_quote(path_str)?)?;
            }
            writeln!(stdout)?;
        }
    }
    Ok(())
}

/// Build and run one or more evals against one or more tools.
fn run_multiple(
    ctrl_c: &mut CtrlC,
    cfg: RunConfig,
    RunRaw {
        eval,
        tool,
        no_eval,
        no_tool,
        dry_run,
    }: RunRaw,
) -> anyhow::Result<Result<(), ExitCode>> {
    let evals = ls("evals")?;
    let (evals_build, evals_run) =
        process_run_items(RunItemKind::Eval, eval, no_eval, || Ok(evals.clone()))?;
    let (tools_build, tools_run) =
        process_run_items(RunItemKind::Tool, tool, no_tool, || ls("tools"))?;
    if dry_run {
        let this = env::args()
            .next()
            .ok_or_else(|| anyhow!("failed to get the name of this program"))?;
        run_dry(
            &mut io::stdout(),
            &this,
            cfg,
            RunItems {
                evals_build: &evals_build,
                tools_build: &tools_build,
                evals_run: &evals_run,
                tools_run: &tools_run,
            },
        )?;
        return Ok(Ok(()));
    }
    let map = evals_to_tools(evals)?;
    let mut eval_outputs = BTreeMap::new();
    let mut tool_outputs = BTreeMap::new();
    let mut need_newline = false;
    for eval in &evals_build {
        println!("{} {eval}", "preparing eval".blue().bold());
        match build_eval(eval, false) {
            Ok(path) => {
                eval_outputs.insert(eval.clone(), path);
                need_newline = true;
            }
            Err(code) => return Ok(Err(code)),
        }
    }
    if need_newline {
        println!();
        need_newline = false;
    }
    for tool in tools_build {
        println!("{} {tool}", "preparing tool".magenta().bold());
        match build_tool(&tool, false) {
            Ok(path) => {
                tool_outputs.insert(tool.clone(), path);
                need_newline = true;
            }
            Err(code) => return Ok(Err(code)),
        }
    }
    if let Some(dir) = &cfg.output {
        fs::create_dir_all(dir)?;
    }
    if need_newline {
        println!();
    }
    if let Some(dir) = &cfg.output {
        for (eval_string, _) in &evals_run {
            fs::create_dir_all(eval_subpath(dir, eval_string))?;
        }
    }
    let mut pass = true;
    let mut first = true;
    let mut evals_run_cmds = Vec::new();
    for (eval_string, eval_cmd) in &evals_run {
        let cmd = match eval_cmd {
            RunCommand::Named { name, args } => {
                let output = eval_outputs
                    .get(name)
                    .ok_or_else(|| anyhow!("missing build output for eval {name:?}"))?;
                let mut cmd = run_cmd(output, args);
                configure_intermediary_subcommand(&mut cmd);
                cmd
            }
            RunCommand::Command(cmd) => {
                let mut cmd = command_from_parts(cmd)
                    .with_context(|| format!("invalid eval command {eval_string:?}"))?;
                configure_intermediary_subcommand(&mut cmd);
                cmd
            }
        };
        evals_run_cmds.push((eval_string.clone(), cmd));
    }
    let mut tools_run_cmds = Vec::new();
    for (tool_string, tool_cmd) in &tools_run {
        let cmd = match tool_cmd {
            RunCommand::Named { name, args } => {
                let output = tool_outputs
                    .get(name)
                    .ok_or_else(|| anyhow!("missing build output for tool {name:?}"))?;
                let mut cmd = run_cmd(output, args);
                configure_intermediary_subcommand(&mut cmd);
                cmd
            }
            RunCommand::Command(cmd) => {
                let mut cmd = command_from_parts(cmd)
                    .with_context(|| format!("invalid tool command {tool_string:?}"))?;
                configure_intermediary_subcommand(&mut cmd);
                cmd
            }
        };
        tools_run_cmds.push((tool_string.clone(), cmd));
    }
    for (eval_string, eval_cmd) in &mut evals_run_cmds {
        let empty = BTreeMap::new();
        let eval_map = map.get(eval_string.as_str()).unwrap_or(&empty);
        for (tool_string, tool_cmd) in &mut tools_run_cmds {
            if !first {
                println!();
            }
            first = false;
            println!(
                "{} {} {eval_string}",
                "running".bold(),
                "eval".blue().bold(),
            );
            println!(
                "{} {} {tool_string}",
                "   with".bold(),
                "tool".magenta().bold(),
            );
            let log_file = cfg
                .output
                .as_ref()
                .map(|dir| fs::File::create(log_subpath(dir, eval_string, tool_string)))
                .transpose()?;
            let outcome = match (eval_cmd.spawn(), tool_cmd.spawn()) {
                (Ok(mut eval_child), Ok(mut tool_child)) => {
                    let result = intermediary::run(
                        ctrl_c,
                        log_file,
                        &mut eval_child,
                        &mut tool_child,
                        cfg.timeout.map(Duration::from_secs),
                    );
                    let _ = eval_child.wait();
                    let _ = tool_child.wait();
                    result
                }
                _ => Err(BadOutcome::Error),
            };
            print!("{} ", " outcome".bold());
            let actual = match outcome {
                Ok(()) => "success",
                Err(BadOutcome::Interrupt) => {
                    println!("interrupt");
                    // This process is about to exit, so don't try to start the next one.
                    return Ok(Ok(()));
                }
                Err(bad_outcome) => <&str>::from(bad_outcome),
            };
            println!("{actual}");
            if cfg.check {
                let expected = eval_map.get(tool_string.as_str()).map(|o| match o {
                    Some(bad_outcome) => <&str>::from(bad_outcome),
                    None => "success",
                });
                match expected {
                    Some(o) => {
                        if actual == o {
                            println!("{} {}", "expected".green().bold(), o.green());
                        } else {
                            println!("{} {}", "expected".red().bold(), o.red());
                            pass = false;
                        }
                    }
                    None => {
                        println!("{} {}", "expected".yellow().bold(), "unknown".yellow());
                        pass = false;
                    }
                };
            }
        }
    }
    Ok(if pass { Ok(()) } else { Err(ExitCode::FAILURE) })
}

/// Print a JSON `value` with a `name` for GitHub Actions.
fn github_output(name: &str, value: impl Serialize) -> anyhow::Result<()> {
    print!("{name}=");
    serde_json::to_writer(io::stdout(), &value)?;
    println!();
    Ok(())
}

/// A map from eval names to the tools that support them.
type Matrix = BTreeMap<String, BTreeMap<Rc<str>, Option<BadOutcome>>>;

/// Return a map from eval names to the tools that support them.
fn evals_to_tools(evals: Vec<String>) -> anyhow::Result<Matrix> {
    let mut map = BTreeMap::new();
    for eval in evals {
        map.insert(eval, BTreeMap::new());
    }
    for result in fs::read_dir("tools")? {
        let entry = result?;
        let tool = Rc::<str>::from(
            entry
                .file_name()
                .into_string()
                .map_err(|name| anyhow!("invalid file name {name:?}"))?,
        );
        for eval_map in map.values_mut() {
            eval_map.insert(Rc::clone(&tool), Some(BadOutcome::Undefined));
        }
        let path = entry.path().join("evals.txt");
        let evals = fs::read_to_string(&path).unwrap_or_default();
        for line in evals.lines() {
            let (eval, outcome) = match line.split_once(' ') {
                None => (line, None),
                Some((eval, outcome)) => {
                    let bad_outcome = BadOutcome::from_str(outcome).with_context(|| {
                        format!("{path:?}: invalid outcome {outcome:?} for eval {eval:?}")
                    })?;
                    (eval, Some(bad_outcome))
                }
            };
            *map.get_mut(eval)
                .ok_or_else(|| anyhow!("eval {eval:?} not found"))?
                .get_mut(&tool)
                .unwrap() = outcome;
        }
    }
    Ok(map)
}

/// A single entry in the `tool` matrix for GitHub Actions.
#[derive(Serialize)]
struct ToolEntry<'a> {
    /// The name of the tool.
    tool: &'a str,

    /// Whether the tool can be built for `linux/arm64`, as opposed to just `linux/amd64`.
    cross: bool,
}

/// A single entry in the `run` matrix for GitHub Actions.
#[derive(Serialize)]
struct RunEntry {
    /// CLI args to pass to the `repo run` subcommand.
    args: String,

    /// The name of the GitHub Actions artifact to produce.
    artifact: String,
}

impl RunEntry {
    fn new(args: String) -> Self {
        let artifact = mangle(&args);
        Self { args, artifact }
    }
}

/// Print the GitHub Actions matrix to stdout.
fn matrix() -> anyhow::Result<()> {
    github_output("date", format!("{}", chrono::Utc::now().format("%Y-%m-%d")))?;
    let mut evals = ls("evals")?;
    evals.sort();
    github_output("eval", &evals)?;
    let mut tools = ls("tools")?;
    tools.sort();
    github_output(
        "tool",
        tools
            .iter()
            .map(|tool| ToolEntry {
                tool,
                cross: tool != "scilean",
            })
            .collect::<Vec<_>>(),
    )?;
    let evals_squish = ["hello", "llsq", "lstm", "particle", "saddle"];
    let mut runs = Vec::new();
    for tool in &tools {
        let mut args = String::new();
        for eval in evals_squish {
            write!(&mut args, "--eval {eval} ")?;
        }
        write!(&mut args, "--tool {tool}")?;
        runs.push(RunEntry::new(args));
    }
    for eval in evals {
        if evals_squish.contains(&eval.as_str()) {
            continue;
        }
        for tool in &tools {
            runs.push(RunEntry::new(format!("--eval {eval} --tool {tool}")));
        }
    }
    let num_jobs = runs.len();
    if num_jobs > 256 {
        bail!("{num_jobs} jobs is too many for the GitHub Actions limit of 256");
    }
    github_output("run", runs.as_slice())?;
    Ok(())
}

/// Run a subcommand from the "Log" command group.
fn log_command(command: LogCommands) -> anyhow::Result<()> {
    match command {
        LogCommands::Trim { input, output } => {
            run_in_out(log::Trim, input.as_deref(), output.as_deref())
        }
        LogCommands::Summary { input } => run_in_out(log::Summary, input.as_deref(), None),
        LogCommands::Flatten { input, output } => log::flatten(&input, &output),
    }
}

/// Run the GradBench CLI, returning a `Result`.
fn cli() -> Result<(), ExitCode> {
    let mut ctrl_c = CtrlC::new().map_err(|error| err_fail(anyhow!(error)))?;
    match Cli::parse().command {
        Commands::Eval { eval, args } => {
            let output = build_eval(&eval, false)?;
            let mut cmd = run_cmd(&output, &args);
            run(&mut cmd)?;
            Ok(())
        }
        Commands::Tool { tool, args } => {
            let output = build_tool(&tool, false)?;
            let mut cmd = run_cmd(&output, &args);
            run(&mut cmd)?;
            Ok(())
        }
        Commands::Run {
            eval,
            tool,
            output,
            timeout,
        } => {
            let log_file = output
                .map(fs::File::create)
                .transpose()
                .map_err(|err| err_fail(anyhow!(err)))?;
            let mut eval_child = shell(&eval)
                .and_then(|mut cmd| {
                    configure_intermediary_subcommand(&mut cmd);
                    Ok(cmd.spawn()?)
                })
                .context("eval")
                .map_err(err_fail)?;
            let mut tool_child = shell(&tool)
                .and_then(|mut cmd| {
                    configure_intermediary_subcommand(&mut cmd);
                    Ok(cmd.spawn()?)
                })
                .context("tool")
                .map_err(err_fail)?;
            let timeout = timeout.map(Duration::from_secs);
            let outcome = intermediary::run(
                &mut ctrl_c,
                log_file,
                &mut eval_child,
                &mut tool_child,
                timeout,
            );
            let eval_wait = eval_child.wait();
            let tool_wait = tool_child.wait();
            match outcome {
                Ok(()) => {
                    if eval_wait.is_ok() && tool_wait.is_ok() {
                        Ok(())
                    } else {
                        Err(ExitCode::FAILURE)
                    }
                }
                Err(bad_outcome) => Err(ExitCode::from(bad_outcome)),
            }
        }
        Commands::ExitCode { outcome } => match BadOutcome::from_str(&outcome) {
            Ok(bad_outcome) => Err(bad_outcome.into()),
            Err(_) => {
                if outcome == "success" {
                    Ok(())
                } else {
                    Err(err_fail(anyhow!("unknown outcome name {outcome:?}")))
                }
            }
        },
        Commands::Repo { command } => {
            check_git()?;
            match command {
                RepoCommands::Run {
                    eval,
                    tool,
                    no_eval,
                    no_tool,
                    output,
                    timeout,
                    check,
                    dry_run,
                } => match run_multiple(
                    &mut ctrl_c,
                    RunConfig {
                        output,
                        timeout,
                        check,
                    },
                    RunRaw {
                        eval,
                        tool,
                        no_eval,
                        no_tool,
                        dry_run,
                    },
                ) {
                    Ok(res) => res,
                    Err(err) => Err(err_fail(err)),
                },
                RepoCommands::Eval { eval, args } => {
                    let output = build_eval(&eval, false)?;
                    let mut cmd = run_cmd(&output, &args);
                    run(&mut cmd)?;
                    Ok(())
                }
                RepoCommands::Tool { tool, args } => {
                    let output = build_tool(&tool, false)?;
                    let mut cmd = run_cmd(&output, &args);
                    run(&mut cmd)?;
                    Ok(())
                }
                RepoCommands::BuildEval { eval } => {
                    build_eval(&eval, false)?;
                    Ok(())
                }
                RepoCommands::BuildTool { tool } => {
                    build_tool(&tool, false)?;
                    Ok(())
                }
                RepoCommands::Lint {
                    fix,
                    clang_format,
                    clippy,
                    eslint,
                    markdown_toc,
                    nixfmt,
                    prettier,
                    ruff_check,
                    ruff_format,
                    runic,
                    rustfmt,
                    typescript,
                } => {
                    let mut lints = lint::Lints::new();
                    lints.flag(clang_format, lint::clang_format);
                    lints.flag(clippy, lint::clippy);
                    lints.flag(eslint, lint::eslint);
                    lints.flag(markdown_toc, lint::markdown_toc);
                    lints.flag(nixfmt, lint::nixfmt);
                    lints.flag(prettier, lint::prettier);
                    lints.flag(ruff_check, lint::ruff_check);
                    lints.flag(ruff_format, lint::ruff_format);
                    lints.flag(runic, lint::runic);
                    lints.flag(rustfmt, lint::rustfmt);
                    lints.flag(typescript, lint::typescript);
                    lints.run(fix)
                }
                RepoCommands::Matrix => matrix().map_err(err_fail),
                RepoCommands::Stats {
                    input,
                    output,
                    date,
                    commit,
                } => {
                    stats::generate(input, output, StatsMetadata { date, commit }).map_err(err_fail)
                }
            }
        }
        Commands::Log { command } => log_command(command).map_err(err_fail),
    }
}

/// Run the GradBench CLI.
pub fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, path::Path, process::ExitCode};

    use goldenfile::Mint;
    use pretty_assertions::assert_eq;
    use strum::IntoEnumIterator;

    use crate::{mangle, process_run_items, run_dry, BadOutcome, RunCommand, RunConfig,
        RunItemKind, RunItems, OUTCOME_HELP};

    #[test]
    fn test_outcome_help() {
        let mut outcome_help = String::from("One of ");
        for outcome in BadOutcome::iter() {
            let s: &str = outcome.into();
            outcome_help.push('`');
            outcome_help.push_str(s);
            outcome_help.push('`');
            outcome_help.push_str(", ");
        }
        outcome_help.push_str("or `success`");
        assert_eq!(OUTCOME_HELP, outcome_help);
    }

    #[test]
    fn test_outcome_exit_codes() {
        let actual: Vec<(BadOutcome, ExitCode)> = BadOutcome::iter()
            .rev()
            .map(|outcome| (outcome, ExitCode::from(outcome)))
            .collect();
        let expected: Vec<(BadOutcome, ExitCode)> = (1..)
            .zip(BadOutcome::iter().rev())
            .map(|(i, outcome)| (outcome, ExitCode::from(i)))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_outcome_error_exit_code_failure() {
        assert_eq!(ExitCode::from(BadOutcome::Error), ExitCode::FAILURE);
    }

    #[test]
    fn test_mangle_empty() {
        assert_eq!(mangle(""), "");
    }

    #[test]
    fn test_mangle_word() {
        assert_eq!(mangle("foo"), "foo");
    }

    #[test]
    fn test_mangle_args() {
        assert_eq!(mangle("foo --bar --baz=qux"), "foo-bar-baz-qux");
    }

    #[test]
    fn test_mangle_cmd() {
        assert_eq!(mangle("$ echo 'an example'"), "echo-an-example");
    }

    fn str_err<T>(s: &str) -> Result<T, String> {
        Err(s.to_string())
    }

    fn strings(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[derive(Debug, PartialEq)]
    enum RunItemCmdSimplified {
        Named { name: String, args: Vec<String> },
        Command(Vec<String>),
    }

    type RunItemSimplified = (String, RunItemCmdSimplified);

    fn run_items<const N: usize>(
        items: [(&str, RunItemCmdSimplified); N],
    ) -> Vec<RunItemSimplified> {
        items
            .into_iter()
            .map(|(string, cmd)| (string.to_string(), cmd))
            .collect()
    }

    fn process_tools(
        items: &[&str],
        omit: &[&str],
        default: &[&str],
    ) -> Result<(Vec<String>, Vec<RunItemSimplified>), String> {
        match process_run_items(RunItemKind::Tool, strings(items), strings(omit), || {
            Ok(strings(default))
        }) {
            Ok((builds, runs)) => Ok((
                builds,
                runs.into_iter()
                    .map(|(string, cmd)| {
                        let cmd = match cmd {
                            RunCommand::Named { name, args } => RunItemCmdSimplified::Named {
                                name,
                                args,
                            },
                            RunCommand::Command(parts) => RunItemCmdSimplified::Command(parts),
                        };
                        (string, cmd)
                    })
                    .collect(),
            )),
            Err(error) => Err(format!("{error:#}")),
        }
    }

    const DEFAULT_EVALS: &[&str] = &["hello", "llsq"];
    const DEFAULT_TOOLS: &[&str] = &["manual", "pytorch", "jax"];

    #[test]
    fn test_run_items_omit() {
        let actual = process_tools(&[], &["jax", "manual", "jax"], DEFAULT_TOOLS);
        let expected = Ok((
            strings(&["pytorch"]),
            run_items([(
                "pytorch",
                RunItemCmdSimplified::Named {
                    name: "pytorch".to_string(),
                    args: Vec::new(),
                },
            )]),
        ));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_args() {
        let actual = process_tools(
            &["manual --bar --baz=qux", "manual --baz=norf"],
            &[],
            DEFAULT_TOOLS,
        );
        let expected = Ok((
            strings(&["manual"]),
            run_items([
                (
                    "manual --bar --baz=qux",
                    RunItemCmdSimplified::Named {
                        name: "manual".to_string(),
                        args: strings(&["--bar", "--baz=qux"]),
                    },
                ),
                (
                    "manual --baz=norf",
                    RunItemCmdSimplified::Named {
                        name: "manual".to_string(),
                        args: strings(&["--baz=norf"]),
                    },
                ),
            ]),
        ));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_cmd() {
        let actual = process_tools(&["$ echo 'an example'"], &[], DEFAULT_TOOLS);
        let expected = Ok((
            strings(&[]),
            run_items([(
                "$ echo 'an example'",
                RunItemCmdSimplified::Command(strings(&["echo", "an example"])),
            )]),
        ));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_conflict() {
        let actual = process_tools(&["manual"], &["manual"], &[]);
        let expected = str_err("`--no-tool` cannot be used together with `--tool`");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_mangled_duplicate() {
        let actual = process_tools(&["manual", "manual"], &[], &[]);
        let expected = str_err("another `--tool` got the same mangled name manual: \"manual\"");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_split_empty() {
        let actual = process_tools(&[""], &[], &[]);
        let expected = str_err("empty `--tool` after splitting: \"\"");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_split_cmd_empty() {
        let actual = process_tools(&["$"], &[], &[]);
        let expected = str_err("empty `--tool` after `$`: \"$\"");
        assert_eq!(actual, expected);
    }

    fn simple_dry_run(stdout: &mut fs::File, evals: &[&str], tools: &[&str], cfg: RunConfig) {
        let (evals_build, evals_run) =
            process_run_items(RunItemKind::Eval, strings(evals), strings(&[]), || {
                Ok(strings(DEFAULT_EVALS))
            })
            .unwrap();
        let (tools_build, tools_run) =
            process_run_items(RunItemKind::Tool, strings(tools), strings(&[]), || {
                Ok(strings(DEFAULT_TOOLS))
            })
            .unwrap();
        run_dry(
            stdout,
            "gradbench",
            cfg,
            RunItems {
                evals_build: &evals_build,
                tools_build: &tools_build,
                evals_run: &evals_run,
                tools_run: &tools_run,
            },
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn test_run_dry() {
        let mut mint = Mint::new("src/outputs");
        let mut stdout = mint.new_goldenfile("dry_run.sh").unwrap();
        simple_dry_run(&mut stdout, &[], &[], RunConfig::default());
    }

    #[cfg(unix)]
    #[test]
    fn test_run_dry_output() {
        use std::path::PathBuf;

        let mut mint = Mint::new("src/outputs");
        let mut stdout = mint.new_goldenfile("dry_run_output.sh").unwrap();
        let cfg = RunConfig {
            output: Some(PathBuf::from("a directory")),
            ..Default::default()
        };
        simple_dry_run(&mut stdout, &[], &[], cfg);
    }

    #[cfg(unix)]
    #[test]
    fn test_run_dry_timeout() {
        let mut mint = Mint::new("src/outputs");
        let mut stdout = mint.new_goldenfile("dry_run_timeout.sh").unwrap();
        let cfg = RunConfig {
            timeout: Some(42),
            ..Default::default()
        };
        simple_dry_run(&mut stdout, &[], &[], cfg);
    }

    fn join_lines(lines: &[&str]) -> String {
        let mut out = String::new();
        for line in lines {
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    #[test]
    fn test_tool_evals_sorted() {
        let dir = Path::new("../../tools");
        let mut mint = Mint::new(dir);
        for entry in fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name();
            let subpath = Path::new(&name).join("evals.txt");
            let Ok(contents) = fs::read_to_string(dir.join(&subpath)) else {
                continue;
            };
            let mut tools: Vec<&str> = contents.lines().collect();
            tools.sort();
            let mut file = mint.new_goldenfile(subpath).unwrap();
            file.write_all(join_lines(&tools).as_bytes()).unwrap();
        }
    }
}
