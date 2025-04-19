use crate::{
    protocol::{EvaluateResponse, LogMessage, LogResponse, Message, StartResponse},
    util::try_read_line,
};

use crate::util::nanostring;
use anyhow::anyhow;
use std::io::{BufRead, Write};
use colored::{Colorize};

pub fn trim(input: &mut impl BufRead, out: &mut impl Write) -> anyhow::Result<()> {
    while let Some(line) = try_read_line(input)? {
        let mut message: LogMessage = serde_json::from_str(&line)?;
        match message.message {
            Message::Evaluate {
                id,
                module,
                function,
                input: _,
                description,
            } => {
                if let Some(response_line) = try_read_line(input)? {
                    let mut response: LogResponse<EvaluateResponse> =
                        serde_json::from_str(&response_line)?;
                    message.message = Message::Evaluate {
                        id,
                        module,
                        function,
                        input: None,
                        description,
                    };
                    response.response.output = None;
                    writeln!(out, "{}", serde_json::to_string(&message)?)?;
                    writeln!(out, "{}", serde_json::to_string(&response)?)?;
                } else {
                    write!(out, "{}", line)?;
                }
            }
            _ => {
                write!(out, "{}", line)?;
                if let Some(response_line) = try_read_line(input)? {
                    write!(out, "{}", response_line)?;
                }
            }
        }
    }
    Ok(())
}

pub fn summary(input: &mut impl BufRead) -> anyhow::Result<()> {
    let mut eval_name = None;
    let mut eval_config = None;
    let mut tool_name = None;
    let mut tool_config = None;
    let mut num_evaluation = 0;
    let mut num_valid = 0;
    let mut num_invalid = 0;
    let mut interrupted = false;
    let mut elapsed_ns = 0;

    // First read the Start message and get the val name.
    if let Some(line) = try_read_line(input)? {
        let message: LogMessage = serde_json::from_str(&line)?;
        match message.message {
            Message::Start { eval, config, .. } => {
                eval_name = eval;
                eval_config = config;
            }
            _ => {
                return Err(anyhow!("invalid log file: expected start message"));
            }
        }
    }

    // Then read the response for the tool name.
    if let Some(line) = try_read_line(input)? {
        let response: LogResponse<StartResponse> = serde_json::from_str(&line)?;
        tool_name = response.response.tool;
        tool_config = response.response.config;
    }

    // Then run through the rest of the messages and collect
    // statistics. Currently we do not do anything with responses,
    // except for noting their 'elapsed' field.
    while let Some(line) = try_read_line(input)? {
        let message: LogMessage = serde_json::from_str(&line)?;
        elapsed_ns = message.elapsed.nanoseconds;
        match message.message {
            Message::Evaluate { .. } => {
                num_evaluation += 1;
            }
            Message::Analysis { valid, .. } => {
                if valid {
                    num_valid += 1;
                } else {
                    num_invalid += 1;
                }
            }
            _ => (),
        }
        // Skip the response.
        if let Some(response_line) = try_read_line(input)? {
            let response: LogResponse<serde_json::Value> = serde_json::from_str(&response_line)?;
            elapsed_ns = response.elapsed.nanoseconds;
        } else {
            interrupted = true;
            break;
        }
    }

    if let Some(eval) = eval_name {
        println!("{:>11}: {}", "eval".blue().bold(), eval)
    } else {
        println!("{:>11}: {}", "eval".blue().bold(), "unknown")
    }
    if let Some(config) = eval_config {
        println!("{:>11}: {}", "config".blue().bold(), config)
    }

    if let Some(tool) = tool_name {
        println!("{:>11}: {}", "tool".magenta().bold(), tool)
    } else {
        println!("{:>11}: unknown", "tool".magenta().bold())
    }
    if let Some(config) = tool_config {
        println!("{:>11}: {}", "config".magenta().bold(), config)
    }

    println!("{:>11}: {}", "evaluations".bold(), num_evaluation);
    println!("{:>11}: {}", "valid".bold(), num_valid);
    println!("{:>11}: {}", "invalid".bold(), num_invalid);
    println!("{:>11}: {}", "elapsed".bold(), nanostring(elapsed_ns));

    if interrupted {
        println!("{}",
                 "Tool did not respond to last evaluation message - this implies crash or timeout.".red())
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::log;
    use goldenfile::Mint;
    use std::io::{BufReader, Cursor, Write};

    fn write_goldenfile(name: &str, bytes: &[u8]) {
        let mut mint = Mint::new("src/outputs");
        let mut file = mint.new_goldenfile(name).unwrap();
        file.write_all(bytes).unwrap();
    }

    #[test]
    fn test_trim_basic() -> anyhow::Result<()> {
        let input = r#"{ "elapsed": { "nanoseconds": 3528846445 }, "message": {"id": 2, "kind": "evaluate", "module": "hello", "function": "square", "input": 1.0} }
{ "elapsed": { "nanoseconds": 3543823169 }, "response": {"id": 2, "success": true, "output": 1.0, "timings": [{"name": "evaluate", "nanoseconds": 0}]} }
"#;
        let input_cursor = Cursor::new(input.as_bytes());
        let mut output: Vec<u8> = Vec::new();
        log::trim(&mut BufReader::new(input_cursor), &mut output)?;
        write_goldenfile("trim_basic.jsonl", &output);
        Ok(())
    }

    #[test]
    fn test_trim_idempotence() -> anyhow::Result<()> {
        let input = r#"{ "elapsed": { "nanoseconds": 3528846445 }, "message": {"id": 2, "kind": "evaluate", "module": "hello", "function": "square"} }
{ "elapsed": { "nanoseconds": 3543823169 }, "response": {"id": 2, "success": true, "timings": [{"name": "evaluate", "nanoseconds": 0}]} }
"#;
        let input_cursor = Cursor::new(input.as_bytes());
        let mut output: Vec<u8> = Vec::new();
        log::trim(&mut BufReader::new(input_cursor), &mut output)?;
        write_goldenfile("trim_idempotence.jsonl", &output);
        Ok(())
    }

    #[test]
    fn test_trim_missing_response() -> anyhow::Result<()> {
        let input = r#"{ "elapsed": { "nanoseconds": 3528846445 }, "message": {"id": 2, "kind": "evaluate", "module": "hello", "function": "square"} }
"#;
        let input_cursor = Cursor::new(input.as_bytes());
        let mut output: Vec<u8> = Vec::new();
        log::trim(&mut BufReader::new(input_cursor), &mut output)?;
        write_goldenfile("trim_missing_response.jsonl", &output);
        Ok(())
    }
}
