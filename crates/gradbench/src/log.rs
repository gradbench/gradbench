use crate::{
    protocol::{EvaluateResponse, LogMessage, LogResponse, Message},
    util::{try_read_line, InOut},
};

use std::io;

pub struct Trim;

impl InOut<anyhow::Result<()>> for Trim {
    fn run(self, read: impl io::Read, mut out: impl io::Write) -> anyhow::Result<()> {
        let input = &mut io::BufReader::new(read);
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
}

#[cfg(test)]
mod tests {
    use crate::{log, util::InOut};
    use goldenfile::Mint;
    use std::io::{Cursor, Write};

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
        log::Trim.run(input_cursor, &mut output)?;
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
        log::Trim.run(input_cursor, &mut output)?;
        write_goldenfile("trim_idempotence.jsonl", &output);
        Ok(())
    }

    #[test]
    fn test_trim_missing_response() -> anyhow::Result<()> {
        let input = r#"{ "elapsed": { "nanoseconds": 3528846445 }, "message": {"id": 2, "kind": "evaluate", "module": "hello", "function": "square"} }
"#;
        let input_cursor = Cursor::new(input.as_bytes());
        let mut output: Vec<u8> = Vec::new();
        log::Trim.run(input_cursor, &mut output)?;
        write_goldenfile("trim_missing_response.jsonl", &output);
        Ok(())
    }
}
