use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context};
use serde::Serialize;

use crate::protocol::Message;

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
    fs::create_dir(&output)
        .with_context(|| format!("error creating stats directory {output:?}"))?;
    let mut table = Vec::new();
    let mut evals = ls("evals")?;
    evals.sort();
    let mut tools = ls("tools")?;
    tools.sort();
    for eval in &evals {
        let mut row = Vec::new();
        for tool in &tools {
            let path = input.join(format!("run-{eval}-{tool}/log.jsonl"));
            let status = summarize(&path).unwrap_or(Status::Unimplemented);
            row.push(Col { tool, status });
        }
        table.push(Row { eval, tools: row });
    }
    let summary = Summary { metadata, table };
    let output = output.join("summary.json");
    let file = fs::File::create(&output)?;
    serde_json::to_writer(file, &summary)?;
    Ok(())
}
