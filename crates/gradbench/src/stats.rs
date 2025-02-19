use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
    rc::Rc,
    time::Duration,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::{
    evals_to_tools, ls,
    protocol::{EvaluateResponse, Message},
    util::nanos_duration,
};

trait CreateFile {
    fn create(&self, subpath: &str, bytes: &[u8]) -> anyhow::Result<()>;
}

impl CreateFile for PathBuf {
    fn create(&self, subpath: &str, bytes: &[u8]) -> anyhow::Result<()> {
        fs::create_dir_all(self)?;
        fs::File::create(self.join(subpath))?.write_all(bytes)?;
        Ok(())
    }
}

trait Scorer<R: BufRead, F: CreateFile> {
    fn score(&mut self, tool: &str, log: R) -> anyhow::Result<f64>;

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

#[derive(Deserialize)]
struct LoggedMessage<T> {
    message: T,
}

fn description(message: Option<Message>) -> Option<String> {
    match message {
        Some(Message::Evaluate { description, .. }) => description,
        _ => None,
    }
}

#[derive(Deserialize)]
struct LoggedResponse<T> {
    response: T,
}

struct ScorerClassic {
    primal: String,
    derivative: String,
    tools: BTreeMap<String, HashMap<Rc<str>, Duration>>,
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
        let mut workloads = HashMap::new();
        let mut message = None;
        for result in log.lines() {
            let line = result?;
            if let Ok(parsed) = serde_json::from_str::<LoggedMessage<Message>>(&line) {
                message = Some(parsed.message);
                continue;
            }
            let msg = message.take();
            if let Ok(parsed) = serde_json::from_str::<LoggedResponse<EvaluateResponse>>(&line) {
                let mut duration = Duration::ZERO;
                for timing in parsed.response.timings.unwrap_or_default() {
                    if timing.name == "evaluate" {
                        duration += nanos_duration(timing.nanoseconds)?;
                    }
                }
                if let Some(description) = description(msg) {
                    *workloads
                        .entry(Rc::from(description))
                        .or_insert(Duration::ZERO) += duration;
                }
            }
        }
        let score = 1. / workloads.values().sum::<Duration>().as_secs_f64();
        if self.tools.insert(tool.to_string(), workloads).is_none() {
            Ok(score)
        } else {
            Err(anyhow::anyhow!("duplicate tool {tool}"))
        }
    }

    fn finish(&self, file: F) -> anyhow::Result<()> {
        file.create("foo.json", &serde_json::to_vec_pretty(&self.tools)?)
    }
}

fn scorer<R: BufRead, F: CreateFile>(eval: &str) -> Box<dyn Scorer<R, F>> {
    match eval {
        "ba" | "gmm" | "ht" | "lstm" => Box::new(ScorerClassic::new("objective", "jacobian")),
        "kmeans" => Box::new(ScorerClassic::new("cost", "dir")),
        _ => Box::new(()),
    }
}

/// A cell in a table of summary data.
#[derive(Debug, Serialize)]
struct Col<'a> {
    /// The name of the tool for this column.
    tool: &'a str,

    /// The score of the tool for this eval.
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
    /// Optional metadata.
    #[serde(flatten)]
    metadata: StatsMetadata,

    /// The table of summary data.
    table: Vec<Row<'a>>,
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
        let mut row = Vec::new();
        let mut scorer = scorer(eval);
        for tool in &tools {
            let score = if supported.contains(tool.as_str()) {
                let path = input.join(format!("run-{eval}-{tool}/log.jsonl"));
                let reader = io::BufReader::new(fs::File::open(&path)?);
                Some(
                    scorer
                        .score(tool, reader)
                        .with_context(|| format!("failed to process {path:?}"))?,
                )
            } else {
                None
            };
            row.push(Col { tool, score });
        }
        scorer.finish(output.join(eval))?;
        table.push(Row { eval, tools: row });
    }
    let summary = Summary { metadata, table };
    let output = output.join("summary.json");
    let file = fs::File::create(&output)?;
    serde_json::to_writer(file, &summary)?;
    Ok(())
}
