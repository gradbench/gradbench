mod intermediary;
mod log;
mod protocol;
mod stats;
mod util;

use std::{
    backtrace::BacktraceStatus,
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    env, fs,
    io::{self, BufRead},
    mem::take,
    path::{Path, PathBuf},
    process::{Command, ExitCode, ExitStatus, Output, Stdio},
    rc::Rc,
    str::FromStr,
    time::Duration,
};

use anyhow::{anyhow, bail, Context};
use clap::{Parser, Subcommand};
use colored::{Color, Colorize};
use itertools::Itertools;
use regex::Regex;
use serde::Serialize;
use stats::StatsMetadata;
use strum::{EnumIter, EnumString, IntoStaticStr};
use util::{stringify_cmd, CtrlC};

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
    /// Run an eval using Docker.
    ///
    /// The Docker image name is `ghcr.io/gradbench/eval-<EVAL>`. If the image is not found locally
    /// (either from being previously downloaded or from being built locally), this command will
    /// first download it from the GitHub Container registry, then run it.
    Eval {
        /// The name of the eval to run
        eval: String,

        /// The Docker image tag, or `latest` by default. For example: `2024-12-01`
        #[clap(short, long)]
        tag: Option<String>,

        /// Docker platform, e.g. `linux/amd64` or `linux/arm64`
        #[clap(long)]
        platform: Option<String>,

        /// Arguments for the eval itself
        args: Vec<String>,
    },

    /// Run a tool using Docker.
    ///
    /// The Docker image name is `ghcr.io/gradbench/tool-<TOOL>`. If the image is not found locally
    /// (either from being previously downloaded or from being built locally), this command will
    /// first download it from the GitHub Container registry, then run it.
    Tool {
        /// The name of the tool to run
        tool: String,

        /// The Docker image tag, or `latest` by default. For example: `2024-12-01`
        #[clap(short, long)]
        tag: Option<String>,

        /// Docker platform, e.g. `linux/amd64` or `linux/arm64`
        #[clap(long)]
        platform: Option<String>,

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
    /// Build and run one or more evals against one of more tools, using Docker.
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

        /// Print commands to stdout instead of running anything
        #[clap(long)]
        dry_run: bool,
    },

    /// Build and run an eval using Docker.
    ///
    /// If the build is not already cached, any output will be printed in blue.
    ///
    /// The Docker image name is `ghcr.io/gradbench/eval-<EVAL>:latest`.
    Eval {
        /// The name of the eval to run
        eval: String,

        /// Docker platform, e.g. `linux/amd64` or `linux/arm64`
        #[clap(long)]
        platform: Option<String>,

        /// Arguments for the eval itself
        args: Vec<String>,
    },

    /// Build and run a tool using Docker.
    ///
    /// If the build is not already cached, any output will be printed in magenta.
    ///
    /// The Docker image name is `ghcr.io/gradbench/tool-<TOOL>:latest`.
    Tool {
        /// The name of the tool to run
        tool: String,

        /// Docker platform, e.g. `linux/amd64` or `linux/arm64`
        #[clap(long)]
        platform: Option<String>,

        /// Arguments for the tool itself
        args: Vec<String>,
    },

    /// Build the Docker image for an eval.
    ///
    /// The Docker image name is `ghcr.io/gradbench/eval-<EVAL>:latest`.
    BuildEval {
        /// The name of the eval to build
        eval: String,

        /// Comma-separated list of Docker platforms to build for, e.g. `linux/amd64,linux/arm64`
        #[clap(long)]
        platform: Option<String>,
    },

    /// Build the Docker image for a tool.
    ///
    /// The Docker image name is `ghcr.io/gradbench/tool-<TOOL>:latest`.
    BuildTool {
        /// The name of the tool to build
        tool: String,

        /// Comma-separated list of Docker platforms to build for, e.g. `linux/amd64,linux/arm64`
        #[clap(long)]
        platform: Option<String>,
    },

    /// Print JSON values for consumption in GitHub Actions.
    ///
    /// Each value is printed on a single line, preceded by the name of that value and an equals
    /// sign. No extra whitespace is printed, because GitHub Actions seems to be sensitive to that.
    Matrix,

    /// Generate summary data files and plots from a directory containing log files.
    ///
    /// The directory should contain a `run-<EVAL>-<TOOL>/log.jsonl` file for each `<EVAL>` under
    /// `evals` and each `<TOOL>` under `tools`.
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
    /// Writes to stdout unless the `--output` option is used. It is
    /// expected that the input log file is well-formed, but not that
    /// it corresponds to a successful run. In particular, the final
    /// message may not have a response - this occurs when the tool
    /// crashes or times out before it gets to respond.
    Trim {
        /// The input log file.
        input: Option<PathBuf>,

        /// The output log file.
        #[clap(short, long)]
        output: Option<PathBuf>,
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

/// Get a command to build the Docker image for an eval.
fn build_eval_cmd(name: &str, platform: Option<&str>) -> Command {
    let mut cmd = Command::new("docker");
    cmd.arg("build");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args([".", "--file"])
        .arg(format!("evals/{name}/Dockerfile"))
        .arg("--tag")
        .arg(format!("ghcr.io/gradbench/eval-{name}"));
    cmd
}

/// Get a command to build the Docker image for a tool.
fn build_tool_cmd(name: &str, platform: Option<&str>) -> Command {
    let mut cmd = Command::new("docker");
    cmd.arg("build");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args([".", "--file"])
        .arg(format!("tools/{name}/Dockerfile"))
        .arg("--tag")
        .arg(format!("ghcr.io/gradbench/tool-{name}"));
    cmd
}

/// Get a command to run an eval using Docker.
fn eval_cmd(name: &str, tag: Option<&str>, platform: Option<&str>, args: &[String]) -> Command {
    let t = tag.unwrap_or("latest");
    let mut cmd = Command::new("docker");
    cmd.arg("run");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args(["--rm", "--interactive"])
        .arg(format!("ghcr.io/gradbench/eval-{name}:{t}"))
        .args(args);
    cmd
}

/// Get a command to run a tool using Docker.
fn tool_cmd(name: &str, tag: Option<&str>, platform: Option<&str>, args: &[String]) -> Command {
    let t = tag.unwrap_or("latest");
    let mut cmd = Command::new("docker");
    cmd.arg("run");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args(["--rm", "--interactive"])
        .arg(format!("ghcr.io/gradbench/tool-{name}:{t}"))
        .args(args);
    cmd
}

/// Run an eval using Docker.
fn run_eval(
    name: &str,
    tag: Option<&str>,
    platform: Option<&str>,
    args: &[String],
) -> Result<(), ExitCode> {
    let mut cmd = eval_cmd(name, tag, platform, args);
    run(&mut cmd)?;
    Ok(())
}

/// Run a tool using Docker.
fn run_tool(
    name: &str,
    tag: Option<&str>,
    platform: Option<&str>,
    args: &[String],
) -> Result<(), ExitCode> {
    let mut cmd = tool_cmd(name, tag, platform, args);
    run(&mut cmd)?;
    Ok(())
}

/// Whether or not Docker output was suppressed due to detected caching.
enum Caching {
    /// Everything seemed to be cached; output was suppressed.
    Cached,

    /// Not everything seemed to be cached; output was not suppressed.
    Uncached,
}

/// Run a `docker build` command but don't print output if everything is cached.
fn docker_build_quiet(color: Color, mut cmd: Command) -> anyhow::Result<(Caching, ExitStatus)> {
    let mut child = cmd
        .arg("--progress=plain")
        // Podman-based Dockers print build logs to stdout, which will
        // interfere with the GradBench protocol when building as part
        // of a 'gradbench repo tool' command. To avoid this, we
        // silence stdout.
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    // A digit mean the start of a number of seconds for an output line for a `RUN` command. The
    // string `sha256` is the start of a line for downloading in a `FROM` command.
    let re = Regex::new(r"^#\d+ (\d|sha256)").unwrap();
    let mut cached = true;
    let mut buffer = String::new();
    colored::control::set_override(true);
    for result in io::BufReader::new(child.stderr.take().unwrap()).lines() {
        let line = result?;
        if cached {
            buffer.push_str(&line);
            buffer.push('\n');
            if re.is_match(&line) {
                cached = false;
                eprint!("{}", take(&mut buffer).color(color));
            }
        } else {
            eprintln!("{}", line.color(color));
        }
    }
    let caching = if cached {
        Caching::Cached
    } else {
        Caching::Uncached
    };
    let status = child.wait()?;
    if !status.success() {
        eprint!("{}", take(&mut buffer).color(color));
    }
    Ok((caching, status))
}

/// A level of verbosity for building a Docker image.
enum Verbosity {
    /// Normal output.
    Normal,

    /// No output except for errors.
    Quiet,
}

/// Build the Docker image for an eval.
fn build_eval(
    name: &str,
    platform: Option<&str>,
    verbosity: Verbosity,
) -> Result<Caching, ExitCode> {
    if name.is_empty() || !fs::exists(Path::new("evals").join(name)).unwrap_or(false) {
        return Err(err_fail(anyhow!("can't find eval to build: {name:?}")));
    }
    let mut cmd = build_eval_cmd(name, platform);
    match verbosity {
        Verbosity::Normal => {
            run(&mut cmd)?;
            Ok(Caching::Uncached)
        }
        Verbosity::Quiet => {
            let (caching, status) = docker_build_quiet(Color::Blue, cmd)
                .with_context(|| format!("error building eval {name}"))
                .map_err(err_fail)?;
            status_code(status)?;
            Ok(caching)
        }
    }
}

/// Build the Docker image for a tool.
fn build_tool(
    name: &str,
    platform: Option<&str>,
    verbosity: Verbosity,
) -> Result<Caching, ExitCode> {
    if name.is_empty() || !fs::exists(Path::new("tools").join(name)).unwrap_or(false) {
        return Err(err_fail(anyhow!("can't find tool to build: {name:?}")));
    }
    let mut cmd = build_tool_cmd(name, platform);
    match verbosity {
        Verbosity::Normal => {
            cmd.arg("--progress=plain");
            run(&mut cmd)?;
            Ok(Caching::Uncached)
        }
        Verbosity::Quiet => {
            let (caching, status) = docker_build_quiet(Color::Magenta, cmd)
                .with_context(|| format!("error building tool {name}"))
                .map_err(err_fail)?;
            status_code(status)?;
            Ok(caching)
        }
    }
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
struct RunConfig {
    /// Evals to run, or all by default.
    eval: Vec<String>,

    /// Tools to run, or all by default.
    tool: Vec<String>,

    /// Evals to omit.
    no_eval: Vec<String>,

    /// Tools to omit.
    no_tool: Vec<String>,

    /// Output directory.
    output: Option<PathBuf>,

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

/// A raw `String` and the processed `Command` representing its semantics.
type RunItem = (String, Command);

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
            let (build, mut cmd) = if first == "$" {
                let program = parts
                    .pop_front()
                    .ok_or_else(|| anyhow!("empty `--{kind}` after `$`: {string:?}"))?;
                let cmd = Command::new(program);
                (None, cmd)
            } else {
                let cmd = (match item_kind {
                    RunItemKind::Eval => eval_cmd,
                    RunItemKind::Tool => tool_cmd,
                })(&first, None, None, &[]);
                (Some(first), cmd)
            };
            for arg in parts {
                cmd.arg(arg);
            }
            configure_intermediary_subcommand(&mut cmd);
            Ok((build, (string, cmd)))
        })
        .process_results::<_, _, anyhow::Error, (BTreeSet<Option<String>>, Vec<RunItem>)>(|it| {
            it.unzip()
        })?;
    Ok((builds.into_iter().flatten().collect(), runs))
}

/// Print the commands for building and running one more evals against one or more tools.
fn run_dry(
    stdout: &mut impl io::Write,
    this: &str,
    output: Option<&Path>,
    evals_build: &[String],
    tools_build: &[String],
    evals_run: &[RunItem],
    tools_run: &[RunItem],
) -> anyhow::Result<()> {
    for eval in evals_build {
        let cmd = shlex::try_join(stringify_cmd(&build_eval_cmd(eval, None))?)?;
        writeln!(stdout, "{cmd}")?;
    }
    for tool in tools_build {
        let cmd = shlex::try_join(stringify_cmd(&build_tool_cmd(tool, None))?)?;
        writeln!(stdout, "{cmd}")?;
    }
    if let Some(dir) = output {
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
            let eval = shlex::try_join(stringify_cmd(eval_cmd)?)?;
            let tool = shlex::try_join(stringify_cmd(tool_cmd)?)?;
            write!(
                stdout,
                "{this} run --eval {} --tool {}",
                shlex::try_quote(&eval)?,
                shlex::try_quote(&tool)?,
            )?;
            if let Some(dir) = output {
                let path = log_subpath(dir, eval_string, tool_string);
                let path_str = path.to_str().ok_or_else(|| {
                    anyhow!("failed to convert output file path to a string: {path:?}")
                })?;
                writeln!(stdout, " -o {}", shlex::try_quote(path_str)?)?;
            } else {
                writeln!(stdout)?;
            }
        }
    }
    Ok(())
}

/// Build and run one or more evals against one or more tools.
fn run_multiple(
    ctrl_c: &mut CtrlC,
    RunConfig {
        eval,
        tool,
        no_eval,
        no_tool,
        output,
        dry_run,
    }: RunConfig,
) -> anyhow::Result<Result<(), ExitCode>> {
    let (evals_build, mut evals_run) =
        process_run_items(RunItemKind::Eval, eval, no_eval, || ls("evals"))?;
    let (tools_build, mut tools_run) =
        process_run_items(RunItemKind::Tool, tool, no_tool, || ls("tools"))?;
    if dry_run {
        let this = env::args()
            .next()
            .ok_or_else(|| anyhow!("failed to get the name of this program"))?;
        run_dry(
            &mut io::stdout(),
            &this,
            output.as_deref(),
            &evals_build,
            &tools_build,
            &evals_run,
            &tools_run,
        )?;
        return Ok(Ok(()));
    }
    let mut need_newline = false;
    for eval in evals_build {
        println!("{} {}", "building eval".blue().bold(), eval);
        match build_eval(&eval, None, Verbosity::Quiet) {
            Ok(Caching::Cached) => need_newline = true,
            Ok(Caching::Uncached) => {
                println!();
                need_newline = false;
            }
            Err(code) => return Ok(Err(code)),
        }
    }
    if need_newline {
        println!();
        need_newline = false;
    }
    for tool in tools_build {
        println!("{} {}", "building tool".magenta().bold(), tool);
        match build_tool(&tool, None, Verbosity::Quiet) {
            Ok(Caching::Cached) => need_newline = true,
            Ok(Caching::Uncached) => {
                println!();
                need_newline = false;
            }
            Err(code) => return Ok(Err(code)),
        }
    }
    if let Some(dir) = &output {
        fs::create_dir_all(dir)?;
    }
    if need_newline {
        println!();
    }
    if let Some(dir) = &output {
        for (eval_string, _) in &evals_run {
            fs::create_dir_all(eval_subpath(dir, eval_string))?;
        }
    }
    let mut first = true;
    for (eval_string, eval_cmd) in &mut evals_run {
        for (tool_string, tool_cmd) in &mut tools_run {
            if !first {
                println!();
            }
            first = false;
            println!(
                "{} {} {}",
                "running".bold(),
                "eval".blue().bold(),
                eval_string,
            );
            println!(
                "{} {} {}",
                "   with".bold(),
                "tool".magenta().bold(),
                tool_string,
            );
            let log_file = output
                .as_ref()
                .map(|dir| fs::File::create(log_subpath(dir, eval_string, tool_string)))
                .transpose()?;
            let outcome = match (eval_cmd.spawn(), tool_cmd.spawn()) {
                (Ok(mut eval_child), Ok(mut tool_child)) => {
                    let result =
                        intermediary::run(ctrl_c, log_file, &mut eval_child, &mut tool_child, None);
                    let _ = eval_child.wait();
                    let _ = tool_child.wait();
                    result
                }
                _ => Err(BadOutcome::Error),
            };
            print!("{} ", "outcome".bold());
            match outcome {
                Ok(()) => println!("success"),
                Err(bad_outcome) => {
                    let stringified: &str = bad_outcome.into();
                    println!("{}", stringified);
                    if let BadOutcome::Interrupt = bad_outcome {
                        // This process is about to exit, so don't try to start the next one.
                        return Ok(Ok(()));
                    }
                }
            }
        }
    }
    Ok(Ok(()))
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
        let tool = entry
            .file_name()
            .into_string()
            .map_err(|name| anyhow!("invalid file name {name:?}"))?;
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
            map.get_mut(eval)
                .ok_or_else(|| anyhow!("eval {eval:?} not found"))?
                .insert(Rc::from(tool.as_str()), outcome);
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
struct RunEntry<'a> {
    /// The name of the eval.
    eval: &'a str,

    /// The name of the tool.
    tool: &'a str,

    /// The expected outcome of the run.
    outcome: &'static str,
}

/// Print the GitHub Actions matrix to stdout.
fn matrix() -> anyhow::Result<()> {
    let date = format!("{}", chrono::Utc::now().format("%Y-%m-%d"));
    github_output("date", date)?;
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
    let mut run = Vec::new();
    let map = evals_to_tools(evals)?;
    for (eval, supported) in &map {
        for tool in &tools {
            run.push(RunEntry {
                eval,
                tool,
                outcome: match supported.get(tool.as_str()) {
                    None => BadOutcome::Undefined.into(),
                    Some(None) => "success",
                    Some(Some(bad_outcome)) => bad_outcome.into(),
                },
            });
        }
    }
    github_output("run", run)?;
    Ok(())
}

/// Run a subcommand from the "Log" command group.
fn log_command(command: LogCommands) -> anyhow::Result<()> {
    match command {
        LogCommands::Trim { input, output } => match (input, output) {
            (Some(input_path), Some(output_path)) => {
                let input_file = fs::File::open(input_path)?;
                let mut output_file = fs::File::create(&output_path)?;
                log::trim(&mut io::BufReader::new(input_file), &mut output_file)?;
                Ok(())
            }
            (Some(input_path), None) => {
                let input_file = fs::File::open(input_path)?;
                log::trim(&mut io::BufReader::new(input_file), &mut io::stdout())?;
                Ok(())
            }
            (None, Some(output_path)) => {
                let mut output_file = fs::File::create(&output_path)?;
                log::trim(&mut io::BufReader::new(io::stdin()), &mut output_file)?;
                Ok(())
            }
            (None, None) => {
                log::trim(&mut io::BufReader::new(io::stdin()), &mut io::stdout())?;
                Ok(())
            }
        },
    }
}

/// Run the GradBench CLI, returning a `Result`.
fn cli() -> Result<(), ExitCode> {
    let mut ctrl_c = CtrlC::new().map_err(|error| err_fail(anyhow!(error)))?;
    match Cli::parse().command {
        Commands::Eval {
            eval,
            tag,
            platform,
            args,
        } => run_eval(&eval, tag.as_deref(), platform.as_deref(), &args),
        Commands::Tool {
            tool,
            tag,
            platform,
            args,
        } => run_tool(&tool, tag.as_deref(), platform.as_deref(), &args),
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
                    dry_run,
                } => match run_multiple(
                    &mut ctrl_c,
                    RunConfig {
                        eval,
                        tool,
                        no_eval,
                        no_tool,
                        output,
                        dry_run,
                    },
                ) {
                    Ok(res) => res,
                    Err(err) => Err(err_fail(err)),
                },
                RepoCommands::Eval {
                    eval,
                    platform,
                    args,
                } => {
                    build_eval(&eval, platform.as_deref(), Verbosity::Quiet)?;
                    run_eval(&eval, None, platform.as_deref(), &args)?;
                    Ok(())
                }
                RepoCommands::Tool {
                    tool,
                    platform,
                    args,
                } => {
                    build_tool(&tool, platform.as_deref(), Verbosity::Quiet)?;
                    run_tool(&tool, None, platform.as_deref(), &args)?;
                    Ok(())
                }
                RepoCommands::BuildEval { eval, platform } => {
                    build_eval(&eval, platform.as_deref(), Verbosity::Normal).map(|_| ())
                }
                RepoCommands::BuildTool { tool, platform } => {
                    build_tool(&tool, platform.as_deref(), Verbosity::Normal).map(|_| ())
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

    use crate::{
        mangle, process_run_items, run_dry, tool_cmd, util::stringify_cmd, BadOutcome, RunItemKind,
        OUTCOME_HELP,
    };

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

    fn simple_tool_cmd(name: &str, args: &[&str]) -> Vec<String> {
        strings(&stringify_cmd(&tool_cmd(name, None, None, &strings(args))).unwrap())
    }

    type RunItemSimplified = (String, Vec<String>);

    fn run_items<const N: usize>(items: [(&str, Vec<String>); N]) -> Vec<RunItemSimplified> {
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
                    .map(|(string, cmd)| (string, strings(&stringify_cmd(&cmd).unwrap())))
                    .collect(),
            )),
            Err(error) => Err(format!("{error:#}")),
        }
    }

    const DEFAULT_EVALS: &[&str] = &["qux", "norf"];
    const DEFAULT_TOOLS: &[&str] = &["foo", "bar", "baz"];

    #[test]
    fn test_run_items_omit() {
        let actual = process_tools(&[], &["baz", "foo", "baz"], DEFAULT_TOOLS);
        let expected = Ok((
            strings(&["bar"]),
            run_items([("bar", simple_tool_cmd("bar", &[]))]),
        ));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_args() {
        let actual = process_tools(
            &["foo --bar --baz=qux", "foo --baz=norf"],
            &[],
            DEFAULT_TOOLS,
        );
        let expected = Ok((
            strings(&["foo"]),
            run_items([
                (
                    "foo --bar --baz=qux",
                    simple_tool_cmd("foo", &["--bar", "--baz=qux"]),
                ),
                ("foo --baz=norf", simple_tool_cmd("foo", &["--baz=norf"])),
            ]),
        ));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_cmd() {
        let actual = process_tools(&["$ echo 'an example'"], &[], DEFAULT_TOOLS);
        let expected = Ok((
            strings(&[]),
            run_items([("$ echo 'an example'", strings(&["echo", "an example"]))]),
        ));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_conflict() {
        let actual = process_tools(&["foo"], &["foo"], &[]);
        let expected = str_err("`--no-tool` cannot be used together with `--tool`");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_run_items_mangled_duplicate() {
        let actual = process_tools(&["foo", "foo"], &[], &[]);
        let expected = str_err("another `--tool` got the same mangled name foo: \"foo\"");
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

    #[test]
    fn test_run_dry() {
        let mut mint = Mint::new("src/outputs");
        let mut stdout = mint.new_goldenfile("dry_run.sh").unwrap();
        let (evals_build, evals_run) =
            process_run_items(RunItemKind::Eval, strings(&[]), strings(&[]), || {
                Ok(strings(DEFAULT_EVALS))
            })
            .unwrap();
        let (tools_build, tools_run) =
            process_run_items(RunItemKind::Tool, strings(&[]), strings(&[]), || {
                Ok(strings(DEFAULT_TOOLS))
            })
            .unwrap();
        run_dry(
            &mut stdout,
            "gradbench",
            Some(Path::new("a directory")),
            &evals_build,
            &tools_build,
            &evals_run,
            &tools_run,
        )
        .unwrap();
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
