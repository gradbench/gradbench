use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Duration,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::{evals_to_tools, ls, protocol::EvaluateResponse, util::nanos_duration};

trait CreateFile {
    fn write(&self, subpath: &str, bytes: &[u8]) -> anyhow::Result<()>;
}

impl CreateFile for PathBuf {
    fn write(&self, subpath: &str, bytes: &[u8]) -> anyhow::Result<()> {
        fs::create_dir_all(self)?;
        fs::File::create(self.join(subpath))?.write_all(bytes)?;
        Ok(())
    }
}

trait Scorer<R: BufRead, F: CreateFile> {
    fn score(&mut self, log: R) -> anyhow::Result<f64>;

    fn finish(&self, file: F) -> anyhow::Result<()>;
}

impl<R: BufRead, F: CreateFile> Scorer<R, F> for () {
    fn score(&mut self, _: R) -> anyhow::Result<f64> {
        Ok(1.)
    }

    fn finish(&self, _: F) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Deserialize)]
struct LoggedResponse<T> {
    response: T,
}

fn score_reciprocal_evaluate_duration(log: impl BufRead) -> anyhow::Result<f64> {
    let mut duration = Duration::ZERO;
    for line in log.lines() {
        if let Ok(parsed) = serde_json::from_str::<LoggedResponse<EvaluateResponse>>(&line?) {
            for timing in parsed.response.timings.unwrap_or_default() {
                if timing.name == "evaluate" {
                    duration += nanos_duration(timing.nanoseconds)?;
                }
            }
        }
    }
    Ok(1. / duration.as_secs_f64())
}

struct ScorerADBench {}

impl ScorerADBench {
    fn new() -> Self {
        Self {}
    }
}

impl<R: BufRead, F: CreateFile> Scorer<R, F> for ScorerADBench {
    fn score(&mut self, log: R) -> anyhow::Result<f64> {
        score_reciprocal_evaluate_duration(log)
    }

    fn finish(&self, _: F) -> anyhow::Result<()> {
        Ok(())
    }
}

struct ScorerKMeans {}

impl ScorerKMeans {
    fn new() -> Self {
        Self {}
    }
}

impl<R: BufRead, F: CreateFile> Scorer<R, F> for ScorerKMeans {
    fn score(&mut self, log: R) -> anyhow::Result<f64> {
        score_reciprocal_evaluate_duration(log)
    }

    fn finish(&self, _: F) -> anyhow::Result<()> {
        Ok(())
    }
}

fn scorer<R: BufRead, F: CreateFile>(eval: &str) -> Box<dyn Scorer<R, F>> {
    match eval {
        "ba" | "gmm" | "ht" | "lstm" => Box::new(ScorerADBench::new()),
        "kmeans" => Box::new(ScorerKMeans::new()),
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
                        .score(reader)
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
