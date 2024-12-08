use std::{
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, ExitStatus, Stdio},
    time::Instant,
};

use anyhow::anyhow;
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
    },

    /// Run a given tool on a given eval. For example:
    ///
    ///     gradbench run -o log.json --eval 'gradbench eval hello' --tool 'gradbench tool pytorch'
    #[clap(about = "Run a given tool on a given eval", verbatim_doc_comment)]
    Run {
        /// A shell script to run the eval. For example: `gradbench eval hello`
        #[clap(long, verbatim_doc_comment)]
        eval: String,

        /// A shell script to run the tool. For example: `gradbench tool pytorch`
        #[clap(long)]
        tool: String,

        /// A path to save the full log. For example: `log.json`
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
    /// Build the Docker image for an eval.
    ///
    /// The Docker image name is `ghcr.io/gradbench/eval-<EVAL>:latest`.
    Eval {
        /// The name of the eval to build
        eval: String,

        /// Build for both `linux/amd64` and `linux/arm64`, instead of just the current architecture
        #[clap(long)]
        cross: bool,
    },

    /// Build the Docker image for a tool.
    ///
    /// The Docker image name is `ghcr.io/gradbench/tool-<TOOL>:latest`.
    Tool {
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
    /// The directory should first contain a `run-<EVAL>-<TOOL>/log.json` file for each `<EVAL>`
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
#[serde(tag = "kind", rename_all = "lowercase")]
enum Message {
    /// A message to define a module.
    Define {
        /// The message ID.
        id: Id,

        /// The name of the module.
        module: String,
    },

    /// A message to evaluate a function.
    Evaluate {
        /// The message ID.
        id: Id,

        /// The name of the function.
        name: String,

        /// A human-readable name for the workload.
        workload: Option<String>,
    },

    /// A final message with all analysis results.
    End,
}

/// A response from the tool to a `"define"` message.
#[derive(Debug, Deserialize)]
struct DefineResponse {
    /// Whether the module was successfully defined.
    success: bool,
}

/// A response from the tool to a `"define"` message.
#[derive(Debug, Deserialize)]
struct EvaluateResponse {
    /// More granular timings.
    nanoseconds: Nanoseconds,
}

/// Nanosecond timings from the tool.
#[derive(Debug, Deserialize)]
struct Nanoseconds {
    /// The time to just evaluate the function.
    evaluate: Option<u128>,
}

/// Analysis from the eval for the tool's response.
#[derive(Debug, Deserialize)]
struct Analysis {
    /// Whether the tool's response was valid.
    correct: bool,

    /// An error message if the tool's response was invalid.
    error: Option<String>,
}

/// Print a message ID with color.
fn tag(id: Id) {
    print!("{}", format!("[{id:>4}]").cyan().bold());
}

/// Run an eval and a tool together, returning the number of validation failures.
fn intermediary(o: &mut impl Write, eval: &mut Child, tool: &mut Child) -> anyhow::Result<usize> {
    let mut invalid = 0;
    let mut eval_in = eval.stdin.take().unwrap();
    let mut tool_in = tool.stdin.take().unwrap();
    let mut eval_out = io::BufReader::new(eval.stdout.take().unwrap());
    let mut tool_out = io::BufReader::new(tool.stdout.take().unwrap());
    writeln!(o, "[")?;
    let mut first = true;
    while let Some(message) = {
        let mut s = String::new();
        s.clear();
        if eval_out.read_line(&mut s)? == 0 {
            None
        } else {
            Some(s)
        }
    } {
        if !first {
            writeln!(o, ",")?;
        }
        first = false;
        write!(o, "  {{\n    \"message\": {}", message.trim())?;
        let message_json: Message = serde_json::from_str(&message)?;
        match &message_json {
            Message::Define { id, module } => {
                writeln!(o, ",")?;
                tag(*id);
                print!(" Defining module {module}... ");
            }
            Message::Evaluate { id, name, workload } => {
                writeln!(o, ",")?;
                tag(*id);
                let workload = workload.as_deref().unwrap_or("");
                print!(" Eval {name:<25} {workload:<15} ");
            }
            Message::End => {
                write!(o, "\n  }}")?;
                break;
            }
        }
        io::stdout().flush()?;
        tool_in.write_all(message.as_bytes())?;
        tool_in.flush()?;
        let start = Instant::now();
        let mut response = String::new();
        tool_out.read_line(&mut response)?;
        let end = Instant::now();
        write!(o, "    \"nanoseconds\": {}", (end - start).as_nanos())?;
        if tool.try_wait()?.is_some() {
            write!(o, "\n  }}")?;
            break;
        }
        write!(o, ",\n    \"response\": {}", response.trim())?;
        eval_in.write_all(response.as_bytes())?;
        eval_in.flush()?;
        match message_json {
            Message::Define { .. } => {
                writeln!(o)?;
                let response_json: DefineResponse = serde_json::from_str(&response)?;
                if response_json.success {
                    println!("Victory!");
                }
            }
            Message::Evaluate { .. } => {
                let response_json: EvaluateResponse = serde_json::from_str(&response)?;
                if let Some(ns) = response_json.nanoseconds.evaluate {
                    print!("{ns:>10} ns ");
                    io::stdout().flush()?;
                }
                let mut analysis = String::new();
                eval_out.read_line(&mut analysis)?;
                writeln!(o, ",\n    \"analysis\": {}", analysis.trim())?;
                let analysis_json: Analysis = serde_json::from_str(&analysis)?;
                if analysis_json.correct {
                    println!("{}", "✓".green());
                } else {
                    println!("{}", "⚠".red());
                    invalid += 1;
                };
                if let Some(error) = analysis_json.error {
                    println!("{}", error.red());
                }
            }
            Message::End => unreachable!(),
        }
        write!(o, "  }}")?;
    }
    writeln!(o, "\n]")?;
    Ok(invalid)
}

/// Return a command's stdout as a string, preserving its exit code whenver possible.
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
#[serde(rename_all = "lowercase")]
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
    let log: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
    let implemented = log.get(0)?.get("response")?.get("success")?;
    if !implemented.as_bool()? {
        return Some(Status::Unimplemented);
    }
    let validations = log.as_array()?.last()?.get("message")?.get("validations")?;
    for validation in validations.as_array()? {
        if !validation.get("correct")?.as_bool()? {
            return Some(Status::Incorrect);
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
        Commands::Eval { eval, tag } => {
            let t = tag.as_deref().unwrap_or("latest");
            run(Command::new("docker")
                .args(["run", "--rm", "--interactive"])
                .arg(format!("ghcr.io/gradbench/eval-{eval}:{t}")))
        }
        Commands::Tool { tool, tag } => {
            let t = tag.as_deref().unwrap_or("latest");
            run(Command::new("docker")
                .args(["run", "--rm", "--interactive"])
                .arg(format!("ghcr.io/gradbench/tool-{tool}:{t}")))
        }
        Commands::Run { eval, tool, output } => {
            let (mut client, mut server) = match (
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
            ) {
                (Ok(e), Ok(t)) => (e, t),
                _ => {
                    eprintln!("error starting eval and tool commands");
                    return Err(ExitCode::FAILURE);
                }
            };
            let result = match output {
                Some(path) => match fs::File::create(&path) {
                    Ok(mut file) => {
                        println!("Writing log to {}.", path.display());
                        intermediary(&mut file, &mut client, &mut server)
                    }
                    Err(err) => Err(anyhow!(err)),
                },
                None => intermediary(&mut io::sink(), &mut client, &mut server),
            };
            match (&result, client.kill(), server.kill()) {
                (&Ok(invalid), Ok(_), Ok(_)) => {
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
                RepoCommands::Eval { eval, cross } => {
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
                RepoCommands::Tool { tool, cross } => {
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
                    let slow = ["scilean"];
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
                            let path = dir.join(format!("run-{eval}-{tool}/log.json"));
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
