use std::{
    fs,
    io::{self, BufRead},
    path::PathBuf,
};

use anyhow::Context;
use serde::Serialize;

use crate::{evals_to_tools, ls};

/// Read a log file and summarize the results.
fn score(reader: impl BufRead) -> anyhow::Result<f64> {
    for res in reader.lines() {
        serde_json::from_str::<serde_json::Value>(&res?)?;
    }
    Ok(1.)
}

/// A cell in a table of summary data.
#[derive(Debug, Serialize)]
struct Col<'a> {
    /// The name of the tool for this column.
    tool: &'a str,

    /// The score of the tool for this eval.
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
    pub date: Option<String>,

    /// The source Git commit SHA.
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
        for tool in &tools {
            let score = if supported.contains(tool.as_str()) {
                let path = input.join(format!("run-{eval}-{tool}/log.jsonl"));
                let reader = io::BufReader::new(fs::File::open(&path)?);
                Some(score(reader).with_context(|| format!("failed to process {path:?}"))?)
            } else {
                None
            };
            row.push(Col { tool, score });
        }
        table.push(Row { eval, tools: row });
    }
    let summary = Summary { metadata, table };
    let output = output.join("summary.json");
    let file = fs::File::create(&output)?;
    serde_json::to_writer(file, &summary)?;
    Ok(())
}
