use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, ExitStatus, Stdio},
    time::{Duration, Instant},
};

use anyhow::anyhow;
use clap::{Parser, Subcommand};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use timeout_readwrite::TimeoutReader;

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

        /// The timeout in seconds for tool responses.
        #[clap(long, default_value_t = 3600)]
        timeout: u64,
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
fn run(command: &mut Command) -> Result<(), ExitCode> {
    let status = command.status().map_err(|err| {
        eprintln!("error running {:?}: {err}", command.get_program());
        ExitCode::FAILURE
    })?;
    status_code(status)
}

/// A message ID.
type Id = i64;

/// A message from the eval.
#[derive(Debug, Deserialize)]
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
        /// The ID of the original message being analyzed.
        of: Id,

        /// Whether the tool's response was valid.
        valid: bool,

        /// An error message if the tool's response was invalid.
        message: Option<String>,
    },
}

/// A response from the tool to a `"start"` message.
#[derive(Debug, Deserialize)]
struct StartResponse {}

/// A response from the tool to a `"define"` message.
#[derive(Debug, Deserialize)]
struct DefineResponse {
    /// Whether the module was successfully defined.
    success: bool,
}

/// Nanosecond timings from the tool.
#[derive(Debug, Deserialize)]
struct Timing {
    /// The name of this timing.
    name: String,

    /// How many nanoseconds elapsed in this timing.
    nanoseconds: u128,
}

/// A response from the tool to an `"evaluate"` message.
#[derive(Debug, Deserialize)]
struct EvaluateResponse {
    /// More granular timings.
    timings: Option<Vec<Timing>>,
}

/// A response from the tool to an `"analysis"` message.
#[derive(Debug, Deserialize)]
struct AnalysisResponse {}

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
    fn start(&mut self, id: Id) {
        if self.id.is_some() {
            self.end();
        }
        print!("{:>WIDTH_ID$}", format!("[{id}]").cyan().bold());
        self.id = Some(id);
    }

    /// Get the current message ID.
    fn id(&self) -> Option<Id> {
        self.id
    }

    /// End a line.
    fn end(&mut self) {
        println!();
        self.id = None;
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

/// Print left-aligned text with a fixed width, preceded by a space.
fn print_left(width: usize, text: &str) {
    if text.len() > width {
        let mut truncated = text.to_string();
        truncated.truncate(width - 3);
        truncated.push_str("...");
        print!(" {truncated:width$}");
    } else {
        print!(" {text:width$}");
    }
}

/// Return a human-readable string for the given number of nanoseconds.
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

/// Print a space, followed by either a green check mark or a red X mark.
fn print_status(success: bool) {
    if success {
        print!(" {}", "✓".green());
    } else {
        print!(" {}", "✗".red());
    }
}

/// Run an eval and a tool together, returning the number of validation failures.
fn intermediary(
    o: &mut impl Write,
    eval: &mut Child,
    tool: &mut Child,
    timeout: Duration,
) -> anyhow::Result<usize> {
    let mut invalid = 0;
    let mut eval_in = eval.stdin.take().unwrap();
    let mut tool_in = tool.stdin.take().unwrap();
    let mut eval_out = io::BufReader::new(eval.stdout.take().unwrap());
    let mut tool_out = io::BufReader::new(TimeoutReader::new(tool.stdout.take().unwrap(), timeout));
    let start = Instant::now();
    let mut line = Line::new();
    while let Some(eval_line) = {
        let mut s = String::new();
        if eval_out.read_line(&mut s)? == 0 {
            None
        } else {
            Some(s)
        }
    } {
        let message_time = Instant::now();
        writeln!(
            o,
            r#"{{ "elapsed": {{ "nanoseconds": {} }}, "message": {} }}"#,
            (message_time - start).as_nanos(),
            eval_line.trim(),
        )?;
        tool_in.write_all(eval_line.as_bytes())?;
        tool_in.flush()?;
        let message: Message = serde_json::from_str(&eval_line)?;
        match &message {
            Message::Start { id: _ } => {
                // Don't print message ID because we're still waiting for the tool to say it's
                // ready, and e.g. if the tool is using `docker run` then it may mess with the
                // terminal output until it actually starts.
            }
            Message::Define { id, module } => {
                line.start(*id);
                print_left(WIDTH_KIND, "def");
                print_left(WIDTH_NAME, module);
            }
            Message::Evaluate {
                id,
                module,
                function,
                input,
                description,
            } => {
                line.start(*id);
                print_left(WIDTH_KIND, "eval");
                let mut workload = match description {
                    Some(s) => s.clone(),
                    None => serde_json::to_string(input)?,
                };
                let width = 15;
                if workload.len() > width {
                    workload.truncate(width - 3);
                    workload.push_str("...");
                }
                print_left(WIDTH_NAME, &format!("{module}::{function}"));
                print_left(WIDTH_DESCRIPTION, &workload);
            }
            Message::Analysis { of, valid, message } => {
                if !*valid {
                    invalid += 1;
                }
                if line.id() == Some(*of) {
                    print_status(*valid);
                    line.end();
                    if let Some(error) = message {
                        println!("{}", error.red());
                    }
                }
            }
        }
        io::stdout().flush()?;
        let mut tool_line = String::new();
        if let Err(err) = tool_out.read_line(&mut tool_line) {
            if err.kind() == io::ErrorKind::TimedOut {
                let ns = (Instant::now() - start).as_nanos();
                writeln!(
                    o,
                    r#"{{ "elapsed": {{ "nanoseconds": {} }}, "response": "timeout" }}"#,
                    ns,
                )?;
                println!("{} {}", nanostring(ns).dimmed(), "⧖".red());
                return Ok(0);
            };
            return Err(anyhow!(err));
        }
        let response_time = Instant::now();
        let nanos = (response_time - message_time).as_nanos();
        writeln!(
            o,
            r#"{{ "elapsed": {{ "nanoseconds": {} }}, "response": {} }}"#,
            (response_time - start).as_nanos(),
            tool_line.trim(),
        )?;
        eval_in.write_all(tool_line.as_bytes())?;
        eval_in.flush()?;
        match message {
            Message::Start { id } => {
                let _: StartResponse = serde_json::from_str(&tool_line)?;
                // OK now that we know the tool won't do anything weird with the terminal.
                line.start(id);
                print_left(WIDTH_KIND, "start");
                line.end();
            }
            Message::Define { .. } => {
                print_left(WIDTH_DESCRIPTION, "");
                print!(" {}", nanostring(nanos).dimmed());
                let response: DefineResponse = serde_json::from_str(&tool_line)?;
                print_status(response.success);
                line.end();
            }
            Message::Evaluate { .. } => {
                print!(" {}", nanostring(nanos).dimmed());
                let response: EvaluateResponse = serde_json::from_str(&tool_line)?;
                let mut timings = BTreeMap::new();
                for Timing { name, nanoseconds } in response.timings.unwrap_or_default() {
                    let (num, ns) = timings.entry(name).or_insert((0, 0));
                    *num += 1;
                    *ns += nanoseconds;
                }
                let mut first = true;
                for (name, (num, ns)) in timings {
                    if first {
                        print!(" {}", "~".dimmed());
                    } else {
                        print!(",");
                    }
                    first = false;
                    print!(" {}", nanostring(ns));
                    print!(" {name}");
                    if num > 1 {
                        print!("×{num}");
                    }
                }
            }
            Message::Analysis { .. } => {}
        }
        io::stdout().flush()?;
    }
    Ok(invalid)
}

/// Return a command's stdout as a string, preserving its exit code whenever possible.
fn stdout(command: &mut Command) -> Result<String, ExitCode> {
    let output = command.output().map_err(|err| {
        eprintln!("error running {:?}: {err}", command.get_program());
        ExitCode::FAILURE
    })?;
    status_code(output.status)?;
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
        if dir.strip_suffix('\n') == cwd.to_str() {
            return Ok(());
        }
    }
    eprintln!(
        "error running a repo subcommand: current working directory is not a Git repository root"
    );
    Err(ExitCode::FAILURE)
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

    /// The table of summary data.
    table: Vec<Row<'a>>,
}

/// Run the GradBench CLI, returning a `Result`.
fn cli_result() -> Result<(), ExitCode> {
    match Cli::parse().command {
        Commands::Eval { eval, tag, args } => {
            let t = tag.as_deref().unwrap_or("latest");
            run(Command::new("docker")
                .args(["run", "--rm", "--interactive"])
                .arg(format!("ghcr.io/gradbench/eval-{eval}:{t}"))
                .args(args))
        }
        Commands::Tool { tool, tag, args } => {
            let t = tag.as_deref().unwrap_or("latest");
            run(Command::new("docker")
                .args(["run", "--rm", "--interactive"])
                .arg(format!("ghcr.io/gradbench/tool-{tool}:{t}"))
                .args(args))
        }
        Commands::Run {
            eval,
            tool,
            output,
            timeout,
        } => {
            let (Ok(mut client), Ok(mut server)) = (
                Command::new("sh")
                    .arg("-c")
                    .arg(eval)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn(),
                Command::new("sh")
                    .arg("-c")
                    .arg(tool)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn(),
            ) else {
                eprintln!("error starting eval and tool commands");
                return Err(ExitCode::FAILURE);
            };
            let timeout = Duration::new(timeout, 0);
            let result = match output {
                Some(path) => match fs::File::create(&path) {
                    Ok(mut file) => intermediary(&mut file, &mut client, &mut server, timeout),
                    Err(err) => Err(anyhow!(err)),
                },
                None => intermediary(&mut io::sink(), &mut client, &mut server, timeout),
            };
            match (&result, client.wait(), server.wait()) {
                (&Ok(invalid), Ok(e), Ok(s)) => {
                    status_code(e)?;
                    status_code(s)?;
                    if invalid == 0 {
                        Ok(())
                    } else {
                        Err(ExitCode::FAILURE)
                    }
                }
                (_, _, _) => {
                    if let Err(err) = result {
                        eprintln!("{err}");
                    }
                    Err(ExitCode::FAILURE)
                }
            }
        }
        Commands::Repo { command } => {
            check_git()?;
            match command {
                RepoCommands::BuildEval { eval, cross } => {
                    let mut cmd = Command::new("docker");
                    cmd.arg("build");
                    if cross {
                        cmd.args(["--platform", "linux/amd64,linux/arm64"]);
                    }
                    cmd.args([".", "--file"])
                        .arg(format!("evals/{eval}/Dockerfile"))
                        .arg("--tag")
                        .arg(format!("ghcr.io/gradbench/eval-{eval}"));
                    run(&mut cmd)
                }
                RepoCommands::BuildTool { tool, cross } => {
                    let mut cmd = Command::new("docker");
                    cmd.arg("build");
                    if cross {
                        cmd.args(["--platform", "linux/amd64,linux/arm64"]);
                    }
                    cmd.args([".", "--file"])
                        .arg(format!("tools/{tool}/Dockerfile"))
                        .arg("--tag")
                        .arg(format!("ghcr.io/gradbench/tool-{tool}"));
                    run(&mut cmd)
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
                RepoCommands::Summarize { dir, date } => {
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
                    let summary = Summary { date, table };
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
pub fn cli() -> ExitCode {
    match cli_result() {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}
