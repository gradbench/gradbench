use std::{
    io::{self, BufRead, Write},
    process::{Child, ChildStdout},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context};
use colored::Colorize;
use indexmap::IndexMap;
use serde::Deserialize;

use crate::{
    err_fail,
    protocol::{
        AnalysisResponse, DefineResponse, EvaluateResponse, Id, Message, StartResponse, Timing,
    },
    util::try_read_line,
    BadOutcome,
};

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

/// An intermediary that runs an eval and a tool, logging their output and timing their execution.
struct Intermediary<IE, IT, OE, OT, C, T, L> {
    outcome: Arc<Mutex<Option<BadOutcome>>>,
    eval_in: IE,
    tool_in: IT,
    eval_out: OE,
    tool_out: OT,
    clock: C,
    out: T,
    log: L,
}

impl<
        IE: Write,
        IT: Write,
        OE: BufRead,
        OT: BufRead,
        C: FnMut() -> Duration,
        T: Write,
        L: Write,
    > Intermediary<IE, IT, OE, OT, C, T, L>
{
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
                let _ = writeln!(self.out, "\n{}", line.red());
            })
            .context("invalid JSON from eval")
    }

    /// Parse a tool response from a line of JSON.
    fn parse_response<'a, R: Deserialize<'a>>(&mut self, line: &'a str) -> anyhow::Result<R> {
        serde_json::from_str(line)
            .inspect_err(|_| {
                let _ = writeln!(self.out, "\n{}", line.red());
            })
            .context("invalid JSON from tool")
    }

    /// Print subtask timings.
    fn print_timings(&mut self, timings: &[Timing]) -> anyhow::Result<()> {
        let mut collected = IndexMap::new();
        for Timing { name, nanoseconds } in timings {
            let (num, ns) = collected.entry(name).or_insert((0, 0));
            *num += 1;
            *ns += nanoseconds;
        }
        let mut first = true;
        for (name, (num, ns)) in collected {
            if first {
                write!(self.out, " {}", "~".dimmed())?;
            } else {
                write!(self.out, ",")?;
            }
            first = false;
            write!(self.out, " {}", nanostring(ns / num))?;
            write!(self.out, " {name}")?;
            if num > 1 {
                write!(self.out, " × {num}")?;
            }
        }
        Ok(())
    }

    /// Run the intermediary, collecting miscellaneous errors via `anyhow`.
    fn run_inner(&mut self) -> anyhow::Result<Option<BadOutcome>> {
        let mut undefined = 0;
        let mut failure = 0;
        let mut invalid = 0;
        let mut line = Line::new();
        while let Some(eval_line) = try_read_line(&mut self.eval_out)? {
            let message_time = (self.clock)();
            writeln!(
                self.log,
                r#"{{ "elapsed": {{ "nanoseconds": {} }}, "message": {} }}"#,
                message_time.as_nanos(),
                eval_line.trim(),
            )?;
            let message: Message = self.parse_message(&eval_line)?;
            match &message {
                Message::Start { .. } => {
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
                    error,
                } => {
                    if !*valid {
                        invalid += 1;
                    }
                    if line.id() == Some(*of) {
                        self.print_status(*valid)?;
                        line.end(&mut self.out)?;
                        if let Some(error) = error {
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
            match self.tool_out.read_line(&mut tool_line) {
                Ok(_) => {}
                Err(err) => {
                    if err.kind() == io::ErrorKind::TimedOut {
                        let timeout_time = (self.clock)();
                        let nanos = (timeout_time - message_time).as_nanos();
                        writeln!(self.out, " {} {}", nanostring(nanos).dimmed(), "⧖".yellow())?;
                        return Ok(Some(BadOutcome::Timeout));
                    } else {
                        return Err(err.into());
                    }
                }
            }
            let response_time = (self.clock)();
            let nanos = (response_time - message_time).as_nanos();
            match message {
                Message::Start { id, eval } => {
                    let response: StartResponse = self.parse_response(&tool_line)?;
                    // OK now that we know the tool won't do anything weird with the terminal.
                    line.start(&mut self.out, id)?;
                    self.print_left(WIDTH_KIND, "start")?;
                    if let Some(name) = eval {
                        write!(self.out, " {name}")?;
                        if let Some(name) = response.tool {
                            write!(self.out, " ({name})")?;
                        }
                    }
                    line.end(&mut self.out)?;
                }
                Message::Define { .. } => {
                    self.print_left(WIDTH_DESCRIPTION, "")?;
                    write!(self.out, " {}", nanostring(nanos).dimmed())?;
                    let response: DefineResponse = self.parse_response(&tool_line)?;
                    if !response.success {
                        undefined += 1;
                    }
                    if let Some(timings) = response.timings {
                        self.print_timings(&timings)?;
                    }
                    self.print_status(response.success)?;
                    line.end(&mut self.out)?;
                    if let Some(error) = response.error {
                        writeln!(self.out, "{}", error.red())?;
                        if response.success {
                            return Err(anyhow!("tool reported success but gave an error"));
                        }
                    }
                }
                Message::Evaluate { .. } => {
                    write!(self.out, " {}", nanostring(nanos).dimmed())?;
                    let response: EvaluateResponse = self.parse_response(&tool_line)?;
                    if !response.success {
                        failure += 1;
                    }
                    if let Some(timings) = response.timings {
                        self.print_timings(&timings)?;
                    }
                    if response.success {
                        if let Some(error) = response.error {
                            line.end(&mut self.out)?;
                            writeln!(self.out, "{}", error.red())?;
                            return Err(anyhow!("tool reported success but gave an error"));
                        } else if response.output.is_none() {
                            line.end(&mut self.out)?;
                            return Err(anyhow!("tool reported success but gave no output"));
                        }
                    } else {
                        self.print_status(false)?;
                        line.end(&mut self.out)?;
                        if let Some(error) = response.error {
                            writeln!(self.out, "{}", error.red())?;
                        }
                    }
                }
                Message::Analysis { .. } => {
                    let _: AnalysisResponse = self.parse_response(&tool_line)?;
                }
            }
            self.out.flush()?;
            // Send the tool's response to the eval only after we've checked that it's valid JSON.
            writeln!(
                self.log,
                r#"{{ "elapsed": {{ "nanoseconds": {} }}, "response": {} }}"#,
                response_time.as_nanos(),
                tool_line.trim(),
            )?;
            self.eval_in.write_all(tool_line.as_bytes())?;
            self.eval_in.flush()?;
        }
        if undefined > 0 {
            Ok(Some(BadOutcome::Undefined))
        } else if failure > 0 {
            Ok(Some(BadOutcome::Failure))
        } else if invalid > 0 {
            Ok(Some(BadOutcome::Invalid))
        } else {
            Ok(None)
        }
    }

    /// Run the intermediary.
    fn run(&mut self) -> Result<(), BadOutcome> {
        let result = self.run_inner();
        if let Some(outcome) = self.outcome.lock().unwrap().take() {
            return Err(outcome);
        }
        match result {
            Ok(None) => Ok(()),
            Ok(Some(outcome)) => Err(outcome),
            Err(err) => {
                let _ = writeln!(self.out, "{}", format!("{err:#}").red());
                Err(BadOutcome::Error)
            }
        }
    }
}

/// Handle Ctrl-C by killing the eval and tool and setting a status flag.
fn handle_ctrlc(
    eval: &mut Child,
    tool: &mut Child,
    outcome: Arc<Mutex<Option<BadOutcome>>>,
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
            *outcome.lock().unwrap() = Some(BadOutcome::Interrupt);
        })?;
    }
    Ok(())
}

/// Return a reader that times out after a given duration, if possible.
fn timeout_reader(reader: ChildStdout, timeout: Option<Duration>) -> impl io::Read {
    #[cfg(unix)]
    {
        timeout_readwrite::TimeoutReader::new(reader, timeout)
    }
    #[cfg(windows)]
    {
        reader
    }
}

/// Run an eval and a tool together, returning the outcome.
pub fn run(
    log: impl Write,
    eval: &mut Child,
    tool: &mut Child,
    timeout: Option<Duration>,
) -> Result<(), BadOutcome> {
    let outcome_mutex = Arc::new(Mutex::new(None));
    match handle_ctrlc(eval, tool, Arc::clone(&outcome_mutex)) {
        Ok(()) => {}
        Err(err) => {
            err_fail(err);
            return Err(BadOutcome::Error);
        }
    }
    let start = Instant::now();
    let outcome = Intermediary {
        outcome: outcome_mutex,
        eval_in: eval.stdin.take().unwrap(),
        tool_in: tool.stdin.take().unwrap(),
        eval_out: io::BufReader::new(eval.stdout.take().unwrap()),
        tool_out: io::BufReader::new(timeout_reader(tool.stdout.take().unwrap(), timeout)),
        clock: || start.elapsed(),
        out: io::stdout(),
        log,
    }
    .run();
    // If fail due to a timeout, the tool may still be running. Kill
    // its process group to ensure that we will not be hanging in a
    // wait() call in main.rs.
    #[cfg(unix)]
    if let Err(BadOutcome::Timeout) = outcome {
        use nix::{sys::signal, unistd};
        if let Ok(tool_id) = tool.id().try_into() {
            let tool_pid = unistd::Pid::from_raw(tool_id);
            if let Ok(pgid) = unistd::getpgid(Some(tool_pid)) {
                let _ = signal::killpg(pgid, signal::Signal::SIGKILL);
            }
        }
    }
    outcome
}

#[cfg(test)]
mod tests {
    use std::{
        f64::consts::{E, PI},
        io::{self, Write},
        sync::{Arc, Mutex},
        time::Duration,
    };

    use goldenfile::Mint;
    use pretty_assertions::assert_eq;
    use serde::{Serialize, Serializer};
    use serde_json::json;

    use crate::intermediary::{
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
            tool: Option<String>,
        },
        Define {
            id: Id,
            success: bool,
            timings: Option<Vec<Timing>>,
            error: Option<String>,
        },
        Evaluate {
            id: Id,
            success: bool,
            output: Option<serde_json::Value>,
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
                Response::Start { id, tool } => StartResponse {
                    id: *id,
                    tool: tool.clone(),
                }
                .serialize(serializer),
                Response::Define {
                    id,
                    success,
                    timings,
                    error,
                } => DefineResponse {
                    id: *id,
                    success: *success,
                    timings: timings.clone(),
                    error: error.clone(),
                }
                .serialize(serializer),
                Response::Evaluate {
                    id,
                    success,
                    output,
                    timings,
                    error,
                } => EvaluateResponse {
                    id: *id,
                    success: *success,
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

    fn write_goldenfile(name: &str, bytes: &[u8]) {
        let mut mint = Mint::new("src/outputs");
        let mut file = mint.new_goldenfile(name).unwrap();
        file.write_all(bytes).unwrap();
    }

    #[test]
    fn test_intermediary_readme_example() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
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
                    success: true,
                    output: Some(json!(E)),
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
                    error: Some("Expected tau, got e.".to_string()),
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
                    success: true,
                    output: Some(json!({"yournumber": 342})),
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
                    error: None,
                },
                Response::Analysis { id: 5 },
            ),
        ]);
        let mut duration = Duration::ZERO;
        let mut increment = Duration::ZERO;
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
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
        write_goldenfile("readme_example.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Invalid));
    }

    #[test]
    fn test_intermediary_start_names() {
        let (eval_out, tool_out) = session(&[(
            Message::Start {
                id: 0,
                eval: Some("foo".to_string()),
            },
            Response::Start {
                id: 0,
                tool: Some("bar".to_string()),
            },
        )]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        let result = intermediary.run();
        write_goldenfile("start_names.txt", &intermediary.out);
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn test_intermediary_define_timings() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: Some(vec![Timing {
                        name: "busywork".to_string(),
                        nanoseconds: Duration::from_millis(10).as_nanos(),
                    }]),
                    error: None,
                },
            ),
        ]);
        let mut duration = Duration::ZERO;
        let mut increment = Duration::ZERO;
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || {
                increment += Duration::from_millis(10);
                duration += increment;
                duration
            },
            out: Vec::new(),
            log: io::sink(),
        };
        let result = intermediary.run();
        write_goldenfile("define_timings.txt", &intermediary.out);
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn test_intermediary_invalid_json_eval() {
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: r#"{ "id": 0,"#.as_bytes(),
            tool_out: "".as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        let result = intermediary.run();
        write_goldenfile("invalid_json_eval.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Error));
    }

    #[test]
    fn test_intermediary_invalid_json_tool() {
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: r#"{ "id": 0, "kind": "start" }"#.as_bytes(),
            tool_out: r#"{ "id": 0,"#.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        let result = intermediary.run();
        write_goldenfile("invalid_json_tool.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Error));
    }

    #[test]
    fn test_intermediary_define_error() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: false,
                    timings: None,
                    error: Some("never heard of foo".to_string()),
                },
            ),
        ]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("define_error.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Undefined));
    }

    #[test]
    fn test_intermediary_define_success_error() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
                    error: Some("all good!".to_string()),
                },
            ),
        ]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("define_success_error.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Error));
    }

    #[test]
    fn test_intermediary_evaluate_error() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
                    error: None,
                },
            ),
            (
                Message::Evaluate {
                    id: 2,
                    module: "foo".to_string(),
                    function: "bar".to_string(),
                    input: json!(42),
                    description: None,
                },
                Response::Evaluate {
                    id: 2,
                    success: false,
                    output: None,
                    timings: None,
                    error: Some("foobar failure".to_string()),
                },
            ),
        ]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("evaluate_error.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Failure));
    }

    #[test]
    fn test_intermediary_evaluate_failure_no_error() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
                    error: None,
                },
            ),
            (
                Message::Evaluate {
                    id: 2,
                    module: "foo".to_string(),
                    function: "bar".to_string(),
                    input: json!(42),
                    description: None,
                },
                Response::Evaluate {
                    id: 2,
                    success: false,
                    output: None,
                    timings: None,
                    error: None,
                },
            ),
        ]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("evaluate_failure_no_error.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Failure));
    }

    #[test]
    fn test_intermediary_evaluate_success_error() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
                    error: None,
                },
            ),
            (
                Message::Evaluate {
                    id: 2,
                    module: "foo".to_string(),
                    function: "bar".to_string(),
                    input: json!(42),
                    description: None,
                },
                Response::Evaluate {
                    id: 2,
                    success: true,
                    output: Some(json!("done")),
                    timings: None,
                    error: Some("foobar all good".to_string()),
                },
            ),
        ]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("evaluate_success_error.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Error));
    }

    #[test]
    fn test_intermediary_evaluate_success_no_output() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
                    error: None,
                },
            ),
            (
                Message::Evaluate {
                    id: 2,
                    module: "foo".to_string(),
                    function: "bar".to_string(),
                    input: json!(42),
                    description: None,
                },
                Response::Evaluate {
                    id: 2,
                    success: true,
                    output: None,
                    timings: None,
                    error: None,
                },
            ),
        ]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("evaluate_success_no_output.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Error));
    }

    #[test]
    fn test_intermediary_evaluate_null_output() {
        let (eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
                    error: None,
                },
            ),
            (
                Message::Evaluate {
                    id: 2,
                    module: "foo".to_string(),
                    function: "null".to_string(),
                    input: json!(null),
                    description: None,
                },
                Response::Evaluate {
                    id: 2,
                    success: true,
                    output: Some(json!(null)),
                    timings: None,
                    error: None,
                },
            ),
            (
                Message::Analysis {
                    id: 3,
                    of: 2,
                    valid: true,
                    error: None,
                },
                Response::Analysis { id: 3 },
            ),
        ]);
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: tool_out.as_bytes(),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("evaluate_success_null_output.txt", &intermediary.out);
        assert_eq!(result, Ok(()));
    }

    struct ReadTimeout<T>(T);

    impl<T: io::Read> io::Read for ReadTimeout<T> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl<T: io::BufRead> io::BufRead for ReadTimeout<T> {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            let result = self.0.fill_buf();
            if let Ok(buf) = &result {
                if buf.is_empty() {
                    return Err(io::Error::new(io::ErrorKind::TimedOut, ""));
                }
            }
            result
        }

        fn consume(&mut self, amt: usize) {
            self.0.consume(amt);
        }
    }

    #[test]
    fn test_intermediary_timeout() {
        let (mut eval_out, tool_out) = session(&[
            (
                Message::Start { id: 0, eval: None },
                Response::Start { id: 0, tool: None },
            ),
            (
                Message::Define {
                    id: 1,
                    module: "foo".to_string(),
                },
                Response::Define {
                    id: 1,
                    success: true,
                    timings: None,
                    error: None,
                },
            ),
        ]);
        eval_out.push_str(
            r#"{ "id": 2, "kind": "evaluate", "module": "foo", "function": "bar", "input": null }"#,
        );
        let mut intermediary = Intermediary {
            outcome: Arc::new(Mutex::new(None)),
            eval_in: io::sink(),
            tool_in: io::sink(),
            eval_out: eval_out.as_bytes(),
            tool_out: ReadTimeout(tool_out.as_bytes()),
            clock: || Duration::ZERO,
            out: Vec::new(),
            log: io::sink(),
        };
        colored::control::set_override(false);
        let result = intermediary.run();
        write_goldenfile("timeout.txt", &intermediary.out);
        assert_eq!(result, Err(BadOutcome::Timeout));
    }
}
