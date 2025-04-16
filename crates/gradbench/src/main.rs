mod intermediary;
mod protocol;
mod stats;
mod util;

use std::{
    backtrace::BacktraceStatus,
    collections::BTreeMap,
    fs,
    io::{self, BufRead},
    mem::take,
    path::{Path, PathBuf},
    process::{Command, ExitCode, ExitStatus, Output, Stdio},
    rc::Rc,
    str::FromStr,
    time::Duration,
};

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use colored::{Color, Colorize};
use regex::Regex;
use serde::Serialize;
use stats::StatsMetadata;
use strum::{EnumIter, EnumString, IntoStaticStr};

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

/// Run an eval using Docker.
fn run_eval(
    name: &str,
    tag: Option<&str>,
    platform: Option<&str>,
    args: &[String],
) -> Result<(), ExitCode> {
    let t = tag.unwrap_or("latest");
    let mut cmd = Command::new("docker");
    cmd.arg("run");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args(["--rm", "--interactive"])
        .arg(format!("ghcr.io/gradbench/eval-{name}:{t}"))
        .args(args);
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
    let t = tag.unwrap_or("latest");
    let mut cmd = Command::new("docker");
    cmd.arg("run");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args(["--rm", "--interactive"])
        .arg(format!("ghcr.io/gradbench/tool-{name}:{t}"))
        .args(args);
    run(&mut cmd)?;
    Ok(())
}

/// A level of verbosity for building a Docker image.
enum Verbosity {
    /// Normal output.
    Normal,

    /// No output except for errors.
    Quiet,
}

/// Run a `docker build` command but don't print output if everything is cached.
fn docker_build_quiet(color: Color, mut cmd: Command) -> anyhow::Result<ExitStatus> {
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
    let status = child.wait()?;
    if !status.success() {
        eprint!("{}", take(&mut buffer).color(color));
    }
    Ok(status)
}

/// Build the Docker image for an eval.
fn build_eval(name: &str, platform: Option<&str>, verbosity: Verbosity) -> Result<(), ExitCode> {
    if name.is_empty() || !fs::exists(Path::new("evals").join(name)).unwrap_or(false) {
        return Err(err_fail(anyhow!("can't find eval to build: {name:?}")));
    }
    let mut cmd = Command::new("docker");
    cmd.arg("build");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args([".", "--file"])
        .arg(format!("evals/{name}/Dockerfile"))
        .arg("--tag")
        .arg(format!("ghcr.io/gradbench/eval-{name}"));
    match verbosity {
        Verbosity::Normal => {
            run(&mut cmd)?;
            Ok(())
        }
        Verbosity::Quiet => status_code(
            docker_build_quiet(Color::Blue, cmd)
                .with_context(|| format!("error building eval {name}"))
                .map_err(err_fail)?,
        ),
    }
}

/// Build the Docker image for a tool.
fn build_tool(name: &str, platform: Option<&str>, verbosity: Verbosity) -> Result<(), ExitCode> {
    if name.is_empty() || !fs::exists(Path::new("tools").join(name)).unwrap_or(false) {
        return Err(err_fail(anyhow!("can't find tool to build: {name:?}")));
    }
    let mut cmd = Command::new("docker");
    cmd.arg("build");
    if let Some(platform) = platform {
        cmd.args(["--platform", platform]);
    }
    cmd.args([".", "--file"])
        .arg(format!("tools/{name}/Dockerfile"))
        .arg("--tag")
        .arg(format!("ghcr.io/gradbench/tool-{name}"));
    match verbosity {
        Verbosity::Normal => {
            cmd.arg("--progress=plain");
            run(&mut cmd)?;
            Ok(())
        }
        Verbosity::Quiet => status_code(
            docker_build_quiet(Color::Magenta, cmd)
                .with_context(|| format!("error building tool {name}"))
                .map_err(err_fail)?,
        ),
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
fn shell(command: &str) -> Command {
    if cfg!(windows) {
        let mut cmd = Command::new("powershell");
        cmd.arg("-Command").arg(command);
        cmd
    } else {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd
    }
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

/// Run the GradBench CLI, returning a `Result`.
fn cli() -> Result<(), ExitCode> {
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
            let (Ok(mut eval_child), Ok(mut tool_child)) = (
                {
                    let mut cmd = shell(&eval);
                    #[cfg(unix)]
                    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
                    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
                },
                {
                    let mut cmd = shell(&tool);
                    #[cfg(unix)]
                    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
                    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
                },
            ) else {
                return Err(err_fail(anyhow!("error starting eval and tool commands")));
            };
            let timeout = timeout.map(Duration::from_secs);
            let outcome = match output {
                Some(path) => {
                    let mut file = fs::File::create(&path).map_err(|err| err_fail(anyhow!(err)))?;
                    intermediary::run(&mut file, &mut eval_child, &mut tool_child, timeout)
                }
                None => {
                    intermediary::run(&mut io::sink(), &mut eval_child, &mut tool_child, timeout)
                }
            };
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
                    build_eval(&eval, platform.as_deref(), Verbosity::Normal)
                }
                RepoCommands::BuildTool { tool, platform } => {
                    build_tool(&tool, platform.as_deref(), Verbosity::Normal)
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

    use crate::{BadOutcome, OUTCOME_HELP};

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
