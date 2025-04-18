use crate::{
    protocol::{EvaluateResponse, LogMessage, LogResponse, Message},
    util::try_read_line,
};

use std::io::{BufRead, Write};

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
