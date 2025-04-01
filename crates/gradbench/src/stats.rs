use std::{
    collections::BTreeMap,
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    rc::Rc,
    time::Duration,
};

use anyhow::{anyhow, bail};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    evals_to_tools, ls,
    protocol::{EvaluateResponse, Message},
    util::nanos_duration,
    BadOutcome,
};

/// Simple trait for creating files inside of an existing directory.
trait CreateFile {
    /// Create a file with the given subpath inside of the directory.
    fn create(&self, subpath: &str, bytes: &[u8]) -> anyhow::Result<()>;
}

impl CreateFile for PathBuf {
    fn create(&self, subpath: &str, bytes: &[u8]) -> anyhow::Result<()> {
        fs::create_dir_all(self)?;
        fs::File::create(self.join(subpath))?.write_all(bytes)?;
        Ok(())
    }
}

/// Trait for scoring multiple tools on an eval by ingesting their log files one-by-one via the type
/// `R: BufRead` and then generating some number of summary files via the type `F: CreateFile`.
trait Scorer<R: BufRead, F: CreateFile> {
    /// Score a `log` file for a `tool`, returning a nonnegative score; higher is better.
    fn score(&mut self, tool: &str, log: R) -> anyhow::Result<f64>;

    /// Finish the scoring process and write the results using `F` to create files if necessary.
    fn finish(&self, file: F) -> anyhow::Result<()>;
}

impl<R: BufRead, F: CreateFile> Scorer<R, F> for () {
    fn score(&mut self, _: &str, _: R) -> anyhow::Result<f64> {
        Ok(1.)
    }

    fn finish(&self, _: F) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A message from the eval to the tool, as wrapped in a log file written by the intermediary.
#[derive(Deserialize)]
struct LoggedMessage<T> {
    /// The message.
    message: T,
}

/// A response from the tool to the eval, as wrapped in a log file written by the intermediary.
#[derive(Deserialize)]
struct LoggedResponse<T> {
    /// The response.
    response: T,
}

/// An average duration for a given eval, tool, and workload, plus the same for the derivative.
#[derive(Clone, Copy, Default, Serialize)]
struct DurationPair {
    /// The average duration to compute the primal value for this workload.
    #[serde(skip_serializing_if = "Option::is_none")]
    primal: Option<Duration>,

    /// The average duration to compute the derivative for this workload.
    #[serde(skip_serializing_if = "Option::is_none")]
    derivative: Option<Duration>,
}

impl DurationPair {
    /// Attempt to set the primal duration.
    fn set_primal(&mut self, primal: Duration) -> anyhow::Result<()> {
        if self.primal.is_some() {
            bail!("primal already set");
        }
        self.primal = Some(primal);
        Ok(())
    }

    /// Attempt to set the derivative duration.
    fn set_derivative(&mut self, derivative: Duration) -> anyhow::Result<()> {
        if self.derivative.is_some() {
            bail!("derivative already set");
        }
        self.derivative = Some(derivative);
        Ok(())
    }

    /// Sum the average duration for the primal with the average duration for the derivative.
    fn sum(self) -> Duration {
        self.primal.unwrap_or(Duration::ZERO) + self.derivative.unwrap_or(Duration::ZERO)
    }
}

/// A scorer for "classic" evals, like the ADBench ones and also _k_-means.
///
/// Each of these evals has exactly two functions, the "primal" function and the "derivative"
/// function. They take the same input but the derivative produces more output.
///
/// The eval should have sent the same set of messages in the same order to each tool, typically a
/// message to evaluate the primal function followed by a message to evaluate the derivative
/// function on the same input.
///
/// For a given message, the tool will run the function some number of times according to the eval's
/// demands; it is expected to include one timing entry with the name `"evaluate"` for each time it
/// ran the function for that message. This scorer averages all those timings within each message.
///
/// The final score is calculated by summing the average time from all messages.
#[derive(Serialize)]
struct ScorerClassic {
    /// The name of the primal function.
    primal: String,

    /// The name of the derivative function.
    derivative: String,

    /// The average duration for each tool (outer keys) and workload (inner keys).
    tools: BTreeMap<String, IndexMap<Rc<str>, DurationPair>>,
}

impl ScorerClassic {
    fn new(primal: impl ToString, derivative: impl ToString) -> Self {
        Self {
            primal: primal.to_string(),
            derivative: derivative.to_string(),
            tools: BTreeMap::new(),
        }
    }
}

impl<R: BufRead, F: CreateFile> Scorer<R, F> for ScorerClassic {
    fn score(&mut self, tool: &str, log: R) -> anyhow::Result<f64> {
        let mut workloads = IndexMap::<Rc<str>, DurationPair>::new();
        let mut message = None;
        for result in log.lines() {
            let line = result?;
            if let Ok(parsed) = serde_json::from_str::<LoggedMessage<Message>>(&line) {
                message = Some(parsed.message);
                continue;
            }
            let Some(msg) = message.take() else {
                bail!("response with no preceding message");
            };
            if let (
                Message::Evaluate {
                    function,
                    description: Some(desc),
                    ..
                },
                Ok(parsed),
            ) = (
                msg,
                serde_json::from_str::<LoggedResponse<EvaluateResponse>>(&line),
            ) {
                let mut count = 0;
                let mut duration = Duration::ZERO;
                for timing in parsed.response.timings.unwrap_or_default() {
                    if timing.name == "evaluate" {
                        count += 1;
                        duration += nanos_duration(timing.nanoseconds)?;
                    }
                }
                // If there are no runs, there's nothing to record here.
                if count > 0 {
                    let pair = workloads.entry(Rc::from(desc)).or_default();
                    if function == self.primal {
                        pair.set_primal(duration / count)?;
                    } else if function == self.derivative {
                        pair.set_derivative(duration / count)?;
                    } else {
                        bail!("unknown function {function:?}");
                    }
                }
            }
        }
        let total = workloads.values().map(|pair| pair.sum()).sum::<Duration>();
        if self.tools.insert(tool.to_string(), workloads).is_none() {
            Ok(1. / total.as_secs_f64())
        } else {
            Err(anyhow!("duplicate tool {tool}"))
        }
    }

    fn finish(&self, file: F) -> anyhow::Result<()> {
        file.create("summary.json", serde_json::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

/// A scorer for evals that contain multiple semantically equivalent
/// but operationally different functions, like particle and saddle.
/// We assume only one workload per function.
///
/// The point of these evals is to investigate the consequences of
/// different implementation choices. In a real setting, one would
/// presumably pick the best one, so the score is reported as the
/// minimum runtime achieved.
#[derive(Serialize)]
struct ScorerEquivFunctions {
    /// The average duration for each tool (outer keys) and function (inner keys).
    tools: BTreeMap<String, IndexMap<Rc<str>, Duration>>,
}

impl ScorerEquivFunctions {
    fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }
}

impl<R: BufRead, F: CreateFile> Scorer<R, F> for ScorerEquivFunctions {
    fn score(&mut self, tool: &str, log: R) -> anyhow::Result<f64> {
        let mut message = None;
        for result in log.lines() {
            let line = result?;
            if let Ok(parsed) = serde_json::from_str::<LoggedMessage<Message>>(&line) {
                message = Some(parsed.message);
                continue;
            }
            let Some(msg) = message.take() else {
                bail!("response with no preceding message");
            };
            if let (
                Message::Evaluate {
                    function,
                    description: _,
                    ..
                },
                Ok(parsed),
            ) = (
                msg,
                serde_json::from_str::<LoggedResponse<EvaluateResponse>>(&line),
            ) {
                let mut count = 0;
                let mut duration = Duration::ZERO;
                for timing in parsed.response.timings.unwrap_or_default() {
                    if timing.name == "evaluate" {
                        count += 1;
                        duration += nanos_duration(timing.nanoseconds)?;
                    }
                }
                self.tools
                    .entry(tool.to_string())
                    .or_default()
                    .insert(Rc::from(function), duration / count);
            }
        }
        if let Some(runtimes) = self.tools.get(tool) {
            let fastest = runtimes
                .values()
                .min()
                .expect("no function worked")
                .as_secs_f64();
            Ok(1.0 / fastest)
        } else {
            Err(anyhow!("no results for tool {tool}"))
        }
    }

    fn finish(&self, file: F) -> anyhow::Result<()> {
        file.create("summary.json", serde_json::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}

/// Return the `Scorer` for the `eval` with the given name.
fn scorer<R: BufRead, F: CreateFile>(eval: &str) -> Box<dyn Scorer<R, F>> {
    match eval {
        "ba" | "gmm" | "ht" | "lstm" => Box::new(ScorerClassic::new("objective", "jacobian")),
        "kmeans" => Box::new(ScorerClassic::new("cost", "dir")),
        "particle" | "saddle" => Box::new(ScorerEquivFunctions::new()),
        "ode" | "llsq" | "det" => Box::new(ScorerClassic::new("primal", "gradient")),
        _ => Box::new(()),
    }
}

/// A cell in a table of summary data.
#[derive(Debug, Serialize)]
struct Col<'a> {
    /// The name of the tool for this column.
    tool: &'a str,

    /// The outcome of the tool for this eval.
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<BadOutcome>,

    /// The score of the tool for this eval, or `None` if the tool was unsuccessful.
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
}

/// A row in a table of summary data.
#[derive(Debug, Serialize)]
struct Row<'a> {
    /// The name of the eval for this row.
    eval: &'a str,

    /// The score of each tool for this eval.
    tools: Vec<Col<'a>>,
}

/// Optional metadata to include in a summary.
#[derive(Debug, Serialize)]
pub struct StatsMetadata {
    /// The current date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,

    /// The source Git commit SHA.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

/// Summary data to be written to a `summary.json` file.
#[derive(Debug, Serialize)]
struct Summary<'a> {
    /// The version of the format for these stats.
    version: usize,

    /// Optional metadata.
    #[serde(flatten)]
    metadata: StatsMetadata,

    /// The table of summary data.
    table: Vec<Row<'a>>,
}

/// Generate a SVG chart for `summary` in the given `output` directory.
fn svg(output: &Path, summary: Summary) -> anyhow::Result<()> {
    let mut file = fs::File::create(output.join("summary.svg"))?;
    let num_evals = summary.table.len() as f64;
    let num_tools = summary.table[0].tools.len() as f64;
    let font_size = 12.;
    let eval_text_length = 60.;
    let tool_text_length = 80.;
    let gap = 5.;
    let cell_size = 30.;
    let total_width = eval_text_length + gap + num_tools * (cell_size + gap);
    let total_height = tool_text_length + gap + num_evals * (cell_size + gap);
    writeln!(
        file,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#,
        total_width, total_height,
    )?;
    writeln!(
        file,
        r##"  <rect x="0" y="0" width="{}" height="{}" rx="{}" ry="{}" fill="#222" />"##,
        total_width, total_height, gap, gap,
    )?;
    for (i, row) in summary.table.iter().enumerate() {
        let x = eval_text_length;
        let y = tool_text_length + gap + cell_size / 2. + i as f64 * (cell_size + gap);
        writeln!(
            file,
            r#"  <text x="{}" y="{}" fill="white" font-family="sans-serif" font-weight="bold" font-size="{}" text-anchor="end" dominant-baseline="middle">{}</text>"#,
            x, y, font_size, row.eval,
        )?;
    }
    for (j, col) in summary.table[0].tools.iter().enumerate() {
        let x = eval_text_length + gap + cell_size / 2. + j as f64 * (cell_size + gap);
        let y = tool_text_length;
        writeln!(
            file,
            r#"  <text x="{}" y="{}" fill="white" font-family="sans-serif" font-weight="bold" font-size="{}" text-anchor="end" dominant-baseline="middle" transform="rotate(90 {} {})">{}</text>"#,
            x, y, font_size, x, y, col.tool,
        )?;
    }
    for (i, row) in summary.table.iter().enumerate() {
        let max_score = row
            .tools
            .iter()
            .filter_map(|col| col.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        for (j, col) in row.tools.iter().enumerate() {
            if col.outcome != Some(BadOutcome::Undefined) {
                let color = match col.score {
                    None => {
                        let alpha = 50;
                        format!("rgb(255 255 255 / {alpha}%)")
                    }
                    Some(score) => {
                        let lightness = 100. - 50. * (score / max_score);
                        format!("hsl(240 100% {lightness}%)")
                    }
                };
                let x = eval_text_length + gap + j as f64 * (cell_size + gap);
                let y = tool_text_length + gap + i as f64 * (cell_size + gap);
                writeln!(
                    file,
                    r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" />"#,
                    x, y, cell_size, cell_size, color,
                )?;
            }
        }
    }
    writeln!(file, "</svg>")?;
    Ok(())
}

/// Generate summary data and plots in `output` from logs in `input`.
pub fn generate(input: PathBuf, output: PathBuf, metadata: StatsMetadata) -> anyhow::Result<()> {
    fs::create_dir_all(&output)?;
    let mut evals = ls("evals")?;
    evals.sort();
    let mut tools = ls("tools")?;
    tools.sort();
    let mut table = Vec::new();
    let map = evals_to_tools(evals)?;
    for (eval, supported) in &map {
        println!("{}", eval);
        let mut row = Vec::new();
        let mut scorer = scorer(eval);
        for tool in &tools {
            let (outcome, score) = match supported.get(tool.as_str()) {
                None => (Some(BadOutcome::Undefined), None),
                Some(&outcome) => {
                    let path = input.join(format!("run-{eval}-{tool}/log.jsonl"));
                    println!("  {}", path.display());
                    let reader = io::BufReader::new(fs::File::open(&path)?);
                    // Always run the `score` method, to gather fine-grained data.
                    let score = scorer.score(tool, reader)?;
                    // Only give the tool an overall score if it successfully completed the eval.
                    (outcome, if outcome.is_none() { Some(score) } else { None })
                }
            };
            row.push(Col {
                tool,
                outcome,
                score,
            });
        }
        scorer.finish(output.join("evals").join(eval))?;
        table.push(Row { eval, tools: row });
    }
    let summary = Summary {
        version: 1,
        metadata,
        table,
    };
    let file = fs::File::create(output.join("summary.json"))?;
    serde_json::to_writer(file, &summary)?;
    svg(&output, summary)?;
    Ok(())
}
