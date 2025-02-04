use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, ExitStatus, Output, Stdio},
    sync::{atomic, Arc},
    time::{Duration, Instant},
};

use anyhow::Context;
use clap::{Parser, Subcommand};
use colored::Colorize;
use serde::{Deserialize, Serialize};

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

        /// Arguments for the tool itself
        args: Vec<String>,
    },

    /// Run a given tool on a given eval. For example:
    ///
    ///     gradbench run -o log.jsonl --eval 'gradbench eval hello' --tool 'gradbench tool pytorch'
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
    /// The Docker image name is `ghcr.io/gradbench/eval-<EVAL>:latest`.
    Eval {
        /// The name of the eval to run
        eval: String,

        /// Arguments for the eval itself
        args: Vec<String>,
    },

    /// Build and run a tool using Docker.
    ///
    /// The Docker image name is `ghcr.io/gradbench/tool-<TOOL>:latest`.
    Tool {
        /// The name of the tool to run
        tool: String,

        /// Arguments for the tool itself
        args: Vec<String>,
    },

    /// Build the Docker image for an eval.
    ///
    /// The Docker image name is `ghcr.io/gradbench/eval-<EVAL>:latest`.
    BuildEval {
        /// The name of the eval to build
        eval: String,

        /// Build for both `linux/amd64` and `linux/arm64`, instead of just the current architecture
        #[clap(long)]
        cross: bool,
    },

    /// Build the Docker image for a tool.
    ///
    /// The Docker image name is `ghcr.io/gradbench/tool-<TOOL>:latest`.
    BuildTool {
        /// The name of the tool to build
        tool: String,

        /// Build for both `linux/amd64` and `linux/arm64`, instead of just the current architecture
        #[clap(long)]
        cross: bool,
    },

    /// Manually build and push a base Docker image. For example:
    ///
    ///     gradbench repo manual mathlib4
    #[clap(
        about = "Manually build and push a base Docker image",
        verbatim_doc_comment
    )]
    Manual {
        /// The image to build and push
        image: String,
    },

    /// Print JSON values for consumption in GitHub Actions.
    ///
    /// Each value is printed on a single line, preceded by the name of that value and an equals
    /// sign. No extra whitespace is printed, because GitHub Actions seems to be sensitive to that.
    Matrix,

    /// Generate summary data files in a directory containing log files.
    ///
    /// The directory should first contain a `run-<EVAL>-<TOOL>/log.jsonl` file for each `<EVAL>`
    /// under `evals` and each `<TOOL>` under `tools`.
    ///
    /// Used in the nightly builds, producing files that can be easily downloaded by JavaScript on
    /// the GradBench website to generate tables and charts.
    Summarize {
        /// The directory to work in
        dir: PathBuf,

        /// The current date
        #[clap(long)]
        date: String,

        /// The Git commit SHA from the `main` branch
        #[clap(long)]
        commit: String,
    },
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
        .map_err(|err| {
            eprintln!("error running {:?}: {err}", command.get_program());
            ExitCode::FAILURE
        })?;
    status_code(output.status)?;
    Ok(output)
}

/// Run an eval using Docker.
fn run_eval(name: &str, tag: Option<&str>, args: &[String]) -> Result<(), ExitCode> {
    let t = tag.unwrap_or("latest");
    run(Command::new("docker")
        .args(["run", "--rm", "--interactive"])
        .arg(format!("ghcr.io/gradbench/eval-{name}:{t}"))
        .args(args))?;
    Ok(())
}

/// Run a tool using Docker.
fn run_tool(name: &str, tag: Option<&str>, args: &[String]) -> Result<(), ExitCode> {
    let t = tag.unwrap_or("latest");
    run(Command::new("docker")
        .args(["run", "--rm", "--interactive"])
        .arg(format!("ghcr.io/gradbench/tool-{name}:{t}"))
        .args(args))?;
    Ok(())
}

/// A set of platforms to build a Docker image for.
enum Platforms {
    /// Build only for the current platform.
    Native,

    /// Build for both x86 and ARM.
    Cross,
}

impl Platforms {
    /// Return a configuration which is cross-platform only if `cross` is `true``.
    fn cross(cross: bool) -> Self {
        if cross {
            Platforms::Cross
        } else {
            Platforms::Native
        }
    }
}

/// A level of verbosity for building a Docker image.
enum Verbosity {
    /// Normal output.
    Normal,

    /// No output except for errors.
    Quiet,
}

/// Build the Docker image for an eval.
fn build_eval(name: &str, platforms: Platforms, verbosity: Verbosity) -> Result<(), ExitCode> {
    let mut cmd = Command::new("docker");
    cmd.arg("build");
    match platforms {
        Platforms::Native => {}
        Platforms::Cross => {
            cmd.args(["--platform", "linux/amd64,linux/arm64"]);
        }
    }
    cmd.args([".", "--file"])
        .arg(format!("evals/{name}/Dockerfile"))
        .arg("--tag")
        .arg(format!("ghcr.io/gradbench/eval-{name}"));
    match verbosity {
        Verbosity::Normal => {}
        Verbosity::Quiet => {
            cmd.arg("--quiet");
            cmd.stdout(Stdio::null()); // Suppress the printed image ID.
        }
    }
    run(&mut cmd)?;
    Ok(())
}

/// Build the Docker image for a tool.
fn build_tool(name: &str, platforms: Platforms, verbosity: Verbosity) -> Result<(), ExitCode> {
    let mut cmd = Command::new("docker");
    cmd.arg("build");
    match platforms {
        Platforms::Native => {}
        Platforms::Cross => {
            cmd.args(["--platform", "linux/amd64,linux/arm64"]);
        }
    }
    cmd.args([".", "--file"])
        .arg(format!("tools/{name}/Dockerfile"))
        .arg("--tag")
        .arg(format!("ghcr.io/gradbench/tool-{name}"));
    match verbosity {
        Verbosity::Normal => {}
        Verbosity::Quiet => {
            cmd.arg("--quiet");
            cmd.stdout(Stdio::null()); // Suppress the printed image ID.
        }
    }
    run(&mut cmd)?;
    Ok(())
}

/// A message ID.
type Id = i64;

/// A message from the eval.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Message {
    /// The first message.
    Start {
        /// The message ID.
        id: Id,
    },

    /// A request to define a module.
    Define {
        /// The message ID.
        id: Id,

        /// The name of the module.
        module: String,
    },

    /// A request to evaluate a function.
    Evaluate {
        /// The message ID.
        id: Id,

        /// The name of the module.
        module: String,

        /// The name of the function.
        function: String,

        /// The input to the function.
        input: serde_json::Value,

        /// A short human-readable description of the input.
        description: Option<String>,
    },

    /// Analysis results from evaluating a function.
    Analysis {
        /// The message ID.
        id: Id,

        /// The ID of the original message being analyzed.
        of: Id,

        /// Whether the tool's response was valid.
        valid: bool,

        /// An error message if the tool's response was invalid.
        message: Option<String>,
    },
}

/// A response from the tool to a `"start"` message.
#[derive(Debug, Deserialize, Serialize)]
struct StartResponse {
    /// The message ID.
    id: Id,
}

/// A response from the tool to a `"define"` message.
#[derive(Debug, Deserialize, Serialize)]
struct DefineResponse {
    /// The message ID.
    id: Id,

    /// Whether the module was successfully defined.
    success: bool,

    /// An error message if the definition failed. Will be None if
    /// the eval is simply not implemented.
    error: Option<String>,
}

/// Nanosecond timings from the tool.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct Timing {
    /// The name of this timing.
    name: String,

    /// How many nanoseconds elapsed in this timing.
    nanoseconds: u128,
}

/// A response from the tool to an `"evaluate"` message.
#[derive(Debug, Deserialize, Serialize)]
struct EvaluateResponse {
    /// The message ID.
    id: Id,

    /// The output of the function.
    output: serde_json::Value,

    /// More granular timings.
    timings: Option<Vec<Timing>>,

    /// An error message if evaluation failed. If this is Some, then
    /// any other fields are not meaningful.
    error: Option<String>,
}

/// A response from the tool to an `"analysis"` message.
#[derive(Debug, Deserialize, Serialize)]
struct AnalysisResponse {
    /// The message ID.
    id: Id,
}

/// A line of printed output.
struct Line {
    /// The message ID of the current line.
    id: Option<Id>,
}

impl Line {
    /// Start with no lines.
    fn new() -> Self {
        Self { id: None }
    }

    /// Begin a new line for a given message ID.
    fn start(&mut self, o: &mut impl Write, id: Id) -> anyhow::Result<()> {
        if self.id.is_some() {
            self.end(o)?;
        }
        write!(o, "{:>WIDTH_ID$}", format!("[{id}]").cyan().bold())?;
        self.id = Some(id);
        Ok(())
    }

    /// Get the current message ID.
    fn id(&self) -> Option<Id> {
        self.id
    }

    /// End a line.
    fn end(&mut self, o: &mut impl Write) -> anyhow::Result<()> {
        writeln!(o)?;
        self.id = None;
        Ok(())
    }
}

/// Width to print the ID of a message, including square brackets.
const WIDTH_ID: usize = 5;

/// Width to print the kind of a message, abbreviated.
const WIDTH_KIND: usize = 5;

/// Width to print the name of a module and function, separated by two colons.
const WIDTH_NAME: usize = 15;

/// Width to print the description of an input.
const WIDTH_DESCRIPTION: usize = 15;

/// Return an 11-character human-readable string for the given number of nanoseconds.
fn nanostring(nanoseconds: u128) -> String {
    let ms = nanoseconds / 1_000_000;
    let sec = ms / 1000;
    let min = sec / 60;
    if sec == 0 {
        format!("{:2} {:2} {:3}ms", "", "", ms)
    } else if min == 0 {
        format!("{:2} {:2}.{:03} s", "", sec, ms % 1000)
    } else if min < 60 {
        format!("{:2}:{:02}.{:03}  ", min, sec % 60, ms % 1000)
    } else {
        format!("{:2} {:2}>{:3}hr", "", "", " 1 ")
    }
}

/// An imperfect outcome from running the intermediary.
#[derive(Debug, Eq, PartialEq)]
enum BadOutcome {
    /// The user sent an interrupt signal.
    Interrupted,

    /// The tool returned some number of invalid results for the eval.
    Invalid,

    /// Some other error occurred. Any relevant information has already been printed.
    Error,
}

/// An intermediary that runs an eval and a tool, logging their output and timing their execution.
struct Intermediary<I, O, C, T, L> {
    interrupted: Arc<atomic::AtomicBool>,
    eval_in: I,
    tool_in: I,
    eval_out: O,
    tool_out: O,
    clock: C,
    out: T,
    log: L,
}

impl<I: Write, O: BufRead, C: FnMut() -> Duration, T: Write, L: Write> Intermediary<I, O, C, T, L> {
    /// Print left-aligned text with a fixed width, preceded by a space.
    fn print_left(&mut self, width: usize, text: &str) -> anyhow::Result<()> {
        if text.len() > width {
            let mut truncated = text.to_string();
            truncated.truncate(width - 3);
            truncated.push_str("...");
            write!(self.out, " {truncated:width$}")?;
        } else {
            write!(self.out, " {text:width$}")?;
        }
        Ok(())
    }

    /// Print a space, followed by either a green check mark or a red X mark.
    fn print_status(&mut self, success: bool) -> anyhow::Result<()> {
        if success {
            write!(self.out, " {}", "✓".green())?;
        } else {
            write!(self.out, " {}", "✗".red())?;
        }
        Ok(())
    }

    /// Parse an eval message from a line of JSON.
    fn parse_message(&mut self, line: &str) -> anyhow::Result<Message> {
        serde_json::from_str(line)
            .inspect_err(|_| {
                let _ = writeln!(self.out, "{line}");
            })
            .context("invalid JSON from eval")
    }

    /// Parse a tool response from a line of JSON.
    fn parse_response<'a, R: Deserialize<'a>>(&mut self, line: &'a str) -> anyhow::Result<R> {
        serde_json::from_str(line)
            .inspect_err(|_| {
                let _ = writeln!(self.out, "{line}");
            })
            .context("invalid JSON from tool")
    }

    /// Run the intermediary, collecting miscellaneous errors via `anyhow`.
    fn run_inner(&mut self) -> anyhow::Result<usize> {
        let mut invalid = 0;
        let mut line = Line::new();
        while let Some(eval_line) = {
            let mut s = String::new();
            if self.eval_out.read_line(&mut s)? == 0 {
                None
            } else {
                Some(s)
            }
        } {
            let message_time = (self.clock)();
            writeln!(
                self.log,
                r#"{{ "elapsed": {{ "nanoseconds": {} }}, "message": {} }}"#,
                message_time.as_nanos(),
                eval_line.trim(),
            )?;
            let message: Message = self.parse_message(&eval_line)?;
            match &message {
                Message::Start { id: _ } => {
                    // Don't print message ID because we're still waiting for the tool to say it's
                    // ready, and e.g. if the tool is using `docker run` then it may mess with the
                    // terminal output until it actually starts.
                }
                Message::Define { id, module } => {
                    line.start(&mut self.out, *id)?;
                    self.print_left(WIDTH_KIND, "def")?;
                    self.print_left(WIDTH_NAME, module)?;
                }
                Message::Evaluate {
                    id,
                    module,
                    function,
                    input,
                    description,
                } => {
                    line.start(&mut self.out, *id)?;
                    self.print_left(WIDTH_KIND, "eval")?;
                    let workload = match description {
                        Some(s) => s.clone(),
                        None => serde_json::to_string(input)?,
                    };
                    self.print_left(WIDTH_NAME, &format!("{module}::{function}"))?;
                    self.print_left(WIDTH_DESCRIPTION, &workload)?;
                }
                Message::Analysis {
                    id: _,
                    of,
                    valid,
                    message,
                } => {
                    if !*valid {
                        invalid += 1;
                    }
                    if line.id() == Some(*of) {
                        self.print_status(*valid)?;
                        line.end(&mut self.out)?;
                        if let Some(error) = message {
                            writeln!(self.out, "{}", error.red())?;
                        }
                    }
                }
            }
            self.out.flush()?;
            // Send the eval's response to the tool only after we've checked that it's valid JSON.
            self.tool_in.write_all(eval_line.as_bytes())?;
            self.tool_in.flush()?;
            let mut tool_line = String::new();
            self.tool_out.read_line(&mut tool_line)?;
            let response_time = (self.clock)();
            let nanos = (response_time - message_time).as_nanos();
            writeln!(
                self.log,
                r#"{{ "elapsed": {{ "nanoseconds": {} }}, "response": {} }}"#,
                response_time.as_nanos(),
                tool_line.trim(),
            )?;
            match message {
                Message::Start { id } => {
                    let _: StartResponse = self.parse_response(&tool_line)?;
                    // OK now that we know the tool won't do anything weird with the terminal.
                    line.start(&mut self.out, id)?;
                    self.print_left(WIDTH_KIND, "start")?;
                    line.end(&mut self.out)?;
                }
                Message::Define { .. } => {
                    self.print_left(WIDTH_DESCRIPTION, "")?;
                    write!(self.out, " {}", nanostring(nanos).dimmed())?;
                    let response: DefineResponse = self.parse_response(&tool_line)?;
                    self.print_status(response.success)?;
                    if let Some(error) = response.error {
                        write!(self.out, "\n{}", error.red());
                        invalid += 1;
                    }
                    line.end(&mut self.out)?;
                }
                Message::Evaluate { .. } => {
                    write!(self.out, " {}", nanostring(nanos).dimmed())?;
                    let response: EvaluateResponse = self.parse_response(&tool_line)?;
                    match response.error {
                        Some(error) => {
                            self.print_status(false)?;
                            print!("\n{}", error.red());
                            invalid += 1;
                        }
                        None => {
                            let mut timings = BTreeMap::new();
                            for Timing { name, nanoseconds } in response.timings.unwrap_or_default()
                            {
                                let (num, ns) = timings.entry(name).or_insert((0, 0));
                                *num += 1;
                                *ns += nanoseconds;
                            }
                            let mut first = true;
                            for (name, (num, ns)) in timings {
                                if first {
                                    write!(self.out, " {}", "~".dimmed())?;
                                } else {
                                    write!(self.out, ",")?;
                                }
                                first = false;
                                write!(self.out, " {}", nanostring(ns))?;
                                write!(self.out, " {name}")?;
                                if num > 1 {
                                    write!(self.out, "×{num}")?;
                                }
                            }
                        }
                    }
                }
                Message::Analysis { .. } => {
                    let _: AnalysisResponse = self.parse_response(&tool_line)?;
                }
            }
            self.out.flush()?;
            // Send the tool's response to the eval only after we've checked that it's valid JSON.
            self.eval_in.write_all(tool_line.as_bytes())?;
            self.eval_in.flush()?;
        }
        Ok(invalid)
    }

    /// Run the intermediary.
    fn run(&mut self) -> Result<(), BadOutcome> {
        let result = self.run_inner();
        if self.interrupted.load(atomic::Ordering::Relaxed) {
            return Err(BadOutcome::Interrupted);
        }
        match result {
            Ok(invalid) => {
                if invalid > 0 {
                    Err(BadOutcome::Invalid)
                } else {
                    Ok(())
                }
            }
            Err(err) => {
                let _ = writeln!(self.out, "{err:#}");
                Err(BadOutcome::Error)
            }
        }
    }
}

/// Handle Ctrl-C by killing the eval and tool and setting a status flag.
fn handle_ctrlc(
    eval: &mut Child,
    tool: &mut Child,
    status: Arc<atomic::AtomicBool>,
) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use nix::{sys::signal, unistd};
        let eval_pid = unistd::Pid::from_raw(eval.id().try_into()?);
        let tool_pid = unistd::Pid::from_raw(tool.id().try_into()?);
        ctrlc::set_handler(move || {
            if let Ok(pgid) = unistd::getpgid(Some(eval_pid)) {
                let _ = signal::killpg(pgid, signal::Signal::SIGKILL);
            }
            if let Ok(pgid) = unistd::getpgid(Some(tool_pid)) {
                let _ = signal::killpg(pgid, signal::Signal::SIGKILL);
            }
            status.store(true, atomic::Ordering::Relaxed);
        })?;
    }
    Ok(())
}

/// Run an eval and a tool together, returning the outcome.
fn intermediary(log: impl Write, eval: &mut Child, tool: &mut Child) -> Result<(), BadOutcome> {
    let interrupted = Arc::new(atomic::AtomicBool::new(false));
    match handle_ctrlc(eval, tool, Arc::clone(&interrupted)) {
        Ok(()) => {}
        Err(err) => {
            println!("{err:#}");
            return Err(BadOutcome::Error);
        }
    }
    let start = Instant::now();
    Intermediary {
        interrupted,
        eval_in: eval.stdin.take().unwrap(),
        tool_in: tool.stdin.take().unwrap(),
        eval_out: io::BufReader::new(eval.stdout.take().unwrap()),
        tool_out: io::BufReader::new(tool.stdout.take().unwrap()),
        clock: || start.elapsed(),
        out: io::stdout(),
        log,
    }
    .run()
}

/// Return a command's stdout as a string, preserving its exit code whenever possible.
fn stdout(command: &mut Command) -> Result<String, ExitCode> {
    let output = run(command.stdout(Stdio::piped()))?;
    String::from_utf8(output.stdout).map_err(|err| {
        let program = command.get_program();
        eprintln!("error processing output from {program:?}: {err}");
        ExitCode::FAILURE
    })
}

/// Check that the current working directory is the root of a Git repository.
fn check_git() -> Result<(), ExitCode> {
    let cwd = env::current_dir().map_err(|err| {
        eprintln!("error getting current working directory: {err}");
        ExitCode::FAILURE
    })?;
    if let Ok(dir) = stdout(Command::new("git").args(["rev-parse", "--show-toplevel"])) {
        if dir.strip_suffix('\n').map(PathBuf::from) == Some(cwd) {
            return Ok(());
        }
    }
    eprintln!(
        "error running a repo subcommand: current working directory is not a Git repository root"
    );
    Err(ExitCode::FAILURE)
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
fn ls(dir: &str) -> Result<Vec<String>, ExitCode> {
    fs::read_dir(dir)
        .map_err(|err| {
            eprintln!("error reading directory {dir:?}: {err}");
            ExitCode::FAILURE
        })?
        .map(|entry| {
            entry
                .map_err(|err| {
                    eprintln!("error reading entry in directory {dir:?}: {err}");
                    ExitCode::FAILURE
                })?
                .file_name()
                .into_string()
                .map_err(|name| {
                    eprintln!("error converting entry name to string: {name:?}");
                    ExitCode::FAILURE
                })
        })
        .collect()
}

/// Print a JSON `value` with a `name` for GitHub Actions.
fn github_output(name: &str, value: impl Serialize) -> Result<(), ExitCode> {
    print!("{name}=");
    serde_json::to_writer(io::stdout(), &value).map_err(|err| {
        eprintln!("error serializing {name}: {err}");
        ExitCode::FAILURE
    })?;
    println!();
    Ok(())
}

/// The status from running a tool on an eval.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    /// The tool is not implemented for the eval.
    Unimplemented,

    /// The tool returned invalid results for the eval.
    Incorrect,

    /// The tool returned correct results for the eval.
    Correct,
}

/// Read a log file and summarize the results.
fn summarize(path: &Path) -> Option<Status> {
    let mut message: Option<Message> = None;
    for res in io::BufReader::new(fs::File::open(path).ok()?).lines() {
        let mut line: serde_json::Value = serde_json::from_str(&res.ok()?).ok()?;
        if let Some(response) = line.get("response") {
            if let Message::Define { .. } = message.take()? {
                if !response.get("success")?.as_bool()? {
                    return Some(Status::Unimplemented);
                }
            }
        }
        if let Some(msg) = line.get_mut("message") {
            message = Some(serde_json::from_value(msg.take()).ok()?);
            if let Some(Message::Analysis { valid, .. }) = &message {
                if !valid {
                    return Some(Status::Incorrect);
                }
            }
        }
    }
    Some(Status::Correct)
}

/// A cell in a table of summary data.
#[derive(Debug, Serialize)]
struct Col<'a> {
    /// The name of the tool for this column.
    tool: &'a str,

    /// The status of the tool for this eval.
    status: Status,
}

/// A row in a table of summary data.
#[derive(Debug, Serialize)]
struct Row<'a> {
    /// The name of the eval for this row.
    eval: &'a str,

    /// The status of each tool for this eval.
    tools: Vec<Col<'a>>,
}

/// Summary data to be written to a `summary.json` file by the `repo summarize` subcommand.
#[derive(Debug, Serialize)]
struct Summary<'a> {
    /// The current date.
    date: String,

    /// The Git commit SHA from the `main` branch.
    commit: String,

    /// The table of summary data.
    table: Vec<Row<'a>>,
}

/// Run the GradBench CLI, returning a `Result`.
fn cli_result() -> Result<(), ExitCode> {
    match Cli::parse().command {
        Commands::Eval { eval, tag, args } => run_eval(&eval, tag.as_deref(), &args),
        Commands::Tool { tool, tag, args } => run_tool(&tool, tag.as_deref(), &args),
        Commands::Run { eval, tool, output } => {
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
                eprintln!("error starting eval and tool commands");
                return Err(ExitCode::FAILURE);
            };
            let outcome = match output {
                Some(path) => match fs::File::create(&path) {
                    Ok(mut file) => intermediary(&mut file, &mut eval_child, &mut tool_child),
                    Err(err) => {
                        println!("{err:#}");
                        return Err(ExitCode::FAILURE);
                    }
                },
                None => intermediary(&mut io::sink(), &mut eval_child, &mut tool_child),
            };
            let eval_wait = eval_child.wait();
            let tool_wait = tool_child.wait();
            match outcome {
                Ok(()) => {
                    status_code(eval_wait.map_err(|_| ExitCode::FAILURE)?)?;
                    status_code(tool_wait.map_err(|_| ExitCode::FAILURE)?)?;
                    Ok(())
                }
                Err(_) => Err(ExitCode::FAILURE),
            }
        }
        Commands::Repo { command } => {
            check_git()?;
            match command {
                RepoCommands::Eval { eval, args } => {
                    build_eval(&eval, Platforms::Native, Verbosity::Quiet)?;
                    run_eval(&eval, None, &args)?;
                    Ok(())
                }
                RepoCommands::Tool { tool, args } => {
                    build_tool(&tool, Platforms::Native, Verbosity::Quiet)?;
                    run_tool(&tool, None, &args)?;
                    Ok(())
                }
                RepoCommands::BuildEval { eval, cross } => {
                    build_eval(&eval, Platforms::cross(cross), Verbosity::Normal)
                }
                RepoCommands::BuildTool { tool, cross } => {
                    build_tool(&tool, Platforms::cross(cross), Verbosity::Normal)
                }
                RepoCommands::Manual { image } => {
                    let name = format!("ghcr.io/gradbench/{image}");
                    run(Command::new("docker")
                        .args(["build", "--platform", "linux/amd64,linux/arm64"])
                        .arg(format!("docker/{image}"))
                        .args(["--tag", &name]))?;
                    let output = stdout(Command::new("docker").args(["run", "--rm", &name]))?;
                    let tag = output.trim();
                    run(Command::new("docker")
                        .args(["tag", &name])
                        .arg(format!("{name}:{tag}")))?;
                    run(Command::new("docker")
                        .arg("push")
                        .arg(format!("{name}:{tag}")))?;
                    Ok(())
                }
                RepoCommands::Matrix => {
                    let date = format!("{}", chrono::Utc::now().format("%Y-%m-%d"));
                    github_output("date", date)?;
                    let mut evals = ls("evals")?;
                    evals.sort();
                    github_output("eval", evals)?;
                    let mut tools = ls("tools")?;
                    tools.retain(|t| t != "diffsharp"); // Flaky.
                    tools.sort();
                    github_output("tool", &tools)?;
                    let slow = ["enzyme", "scilean"];
                    let fast: Vec<_> = tools
                        .iter()
                        .filter(|t| !slow.contains(&t.as_str()))
                        .collect();
                    github_output("fast", fast)?;
                    github_output("slow", slow)?;
                    Ok(())
                }
                RepoCommands::Summarize { dir, date, commit } => {
                    let mut table = vec![];
                    let mut evals = ls("evals")?;
                    evals.sort();
                    let mut tools = ls("tools")?;
                    tools.sort();
                    for eval in &evals {
                        let mut row = vec![];
                        for tool in &tools {
                            let path = dir.join(format!("run-{eval}-{tool}/log.jsonl"));
                            let status = summarize(&path).unwrap_or(Status::Unimplemented);
                            row.push(Col { tool, status });
                        }
                        table.push(Row { eval, tools: row });
                    }
                    let summary = Summary {
                        date,
                        commit,
                        table,
                    };
                    let output = dir.join("summary.json");
                    let file = fs::File::create(&output).map_err(|err| {
                        eprintln!("error creating summary file {output:?}: {err}");
                        ExitCode::FAILURE
                    })?;
                    serde_json::to_writer(file, &summary).map_err(|err| {
                        eprintln!("error serializing summary table: {err}");
                        ExitCode::FAILURE
                    })?;
                    Ok(())
                }
            }
        }
    }
}

/// Run the GradBench CLI.
pub fn main() -> ExitCode {
    match cli_result() {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        f64::consts::{E, PI},
        io::{self, Write},
        sync::{atomic, Arc},
        time::Duration,
    };

    use goldenfile::Mint;
    use serde::{Serialize, Serializer};
    use serde_json::json;

    use crate::{
        nanostring, AnalysisResponse, BadOutcome, DefineResponse, EvaluateResponse, Id,
        Intermediary, Message, StartResponse, Timing,
    };

    fn nanostring_test(expected: &str, duration: Duration) {
        assert_eq!(expected.len(), 11);
        assert_eq!(nanostring(duration.as_nanos()), expected);
    }

    #[test]
    fn test_nanostring_0() {
        nanostring_test("        0ms", Duration::ZERO);
    }

    #[test]
    fn test_nanostring_999_microseconds() {
        nanostring_test("        0ms", Duration::from_micros(999));
    }

    #[test]
    fn test_nanostring_1_millisecond() {
        nanostring_test("        1ms", Duration::from_millis(1));
    }

    #[test]
    fn test_nanostring_999_milliseconds() {
        nanostring_test("      999ms", Duration::from_millis(999));
    }

    #[test]
    fn test_nanostring_1_second() {
        nanostring_test("    1.000 s", Duration::from_secs(1));
    }

    #[test]
    fn test_nanostring_59_seconds() {
        nanostring_test("   59.000 s", Duration::from_secs(59));
    }

    #[test]
    fn test_nanostring_1_minute() {
        nanostring_test(" 1:00.000  ", Duration::from_secs(60));
    }

    #[test]
    fn test_nanostring_59_minutes() {
        nanostring_test("59:00.000  ", Duration::from_secs(59 * 60));
    }

    #[test]
    fn test_nanostring_1_hour() {
        nanostring_test("     > 1 hr", Duration::from_secs(3600));
    }

    enum Response {
        Start {
            id: Id,
        },
        Define {
            id: Id,
            success: bool,
            error: Option<String>,
        },
        Evaluate {
            id: Id,
            output: serde_json::Value,
            timings: Option<Vec<Timing>>,
            error: Option<String>,
        },
        Analysis {
            id: Id,
        },
    }

    impl Serialize for Response {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            match self {
                &Response::Start { id } => StartResponse { id }.serialize(serializer),
                Response::Define { id, success, error } => DefineResponse {
                    id: *id,
                    success: *success,
                    error: error.clone(),
                }
                .serialize(serializer),
                Response::Evaluate {
                    id,
                    output,
                    timings,
                    error,
                } => EvaluateResponse {
                    id: *id,
                    output: output.clone(),
                    timings: timings.clone(),
                    error: error.clone(),
                }
                .serialize(serializer),
                &Response::Analysis { id } => AnalysisResponse { id }.serialize(serializer),
            }
        }
    }

    fn session(pairs: &[(Message, Response)]) -> (String, String) {
        let mut eval_out = Vec::new();
        let mut tool_out = Vec::new();
        for (message, response) in pairs {
            serde_json::to_writer(&mut eval_out, message).unwrap();
            eval_out.extend_from_slice(b"\n");
            serde_json::to_writer(&mut tool_out, response).unwrap();
            tool_out.extend_from_slice(b"\n");
        }
        (
            String::from_utf8(eval_out).unwrap(),
            String::from_utf8(tool_out).unwrap(),
        )
    }

    #[test]
    fn test_intermediary_readme_example() {
        let (eval_out, tool_out) = session(&[
            (Message::Start { id: 0 }, Response::Start { id: 0 }),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    error: None,
                },
            ),
            (
                Message::Evaluate {
                    id: 2,
                    module: "foo".to_string(),
                    function: "bar".to_string(),
                    input: json!(PI),
                    description: None,
                },
                Response::Evaluate {
                    id: 2,
                    output: json!(E),
                    timings: Some(vec![Timing {
                        name: "evaluate".to_string(),
                        nanoseconds: Duration::from_millis(5).as_nanos(),
                    }]),
                    error: None,
                },
            ),
            (
                Message::Analysis {
                    id: 3,
                    of: 2,
                    valid: false,
                    message: Some("Expected tau, got e.".to_string()),
                },
                Response::Analysis { id: 3 },
            ),
            (
                Message::Evaluate {
                    id: 4,
                    module: "foo".to_string(),
                    function: "baz".to_string(),
                    input: json!({"mynumber": 121}),
                    description: None,
                },
                Response::Evaluate {
                    id: 4,
                    output: json!({"yournumber": 342}),
                    timings: Some(vec![Timing {
                        name: "evaluate".to_string(),
                        nanoseconds: Duration::from_millis(7).as_nanos(),
                    }]),
                    error: None,
                },
            ),
            (
                Message::Analysis {
                    id: 5,
                    of: 4,
                    valid: true,
                    message: None,
                },
                Response::Analysis { id: 5 },
            ),
        ]);
        let mut duration = Duration::ZERO;
        let mut increment = Duration::ZERO;
        let mut intermediary = Intermediary {
            interrupted: Arc::new(atomic::AtomicBool::new(false)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || {
                increment += Duration::from_millis(1);
                duration += increment;
                duration
            },
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        {
            let mut mint = Mint::new("src/outputs");
            let mut file = mint.new_goldenfile("readme_example.txt").unwrap();
            file.write_all(&intermediary.out).unwrap();
        }
        assert_eq!(result, Err(BadOutcome::Invalid));
    }

    #[test]
    fn test_intermediary_invalid_json_eval() {
        let mut intermediary = Intermediary {
            interrupted: Arc::new(atomic::AtomicBool::new(false)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: "{ \"id\": 0,".as_bytes(),
            tool_out: "".as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        let result = intermediary.run();
        {
            let mut mint = Mint::new("src/outputs");
            let mut file = mint.new_goldenfile("invalid_json_eval.txt").unwrap();
            file.write_all(&intermediary.out).unwrap();
        }
        assert_eq!(result, Err(BadOutcome::Error));
    }

    #[test]
    fn test_intermediary_invalid_json_tool() {
        let mut intermediary = Intermediary {
            interrupted: Arc::new(atomic::AtomicBool::new(false)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: "{ \"id\": 0, \"kind\": \"start\" }".as_bytes(),
            tool_out: "{ \"id\": 0,".as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        let result = intermediary.run();
        {
            let mut mint = Mint::new("src/outputs");
            let mut file = mint.new_goldenfile("invalid_json_tool.txt").unwrap();
            file.write_all(&intermediary.out).unwrap();
        }
        assert_eq!(result, Err(BadOutcome::Error));
    }
}
