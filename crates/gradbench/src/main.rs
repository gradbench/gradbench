mod parse;

use std::fs;

use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::error::RichReason;
use clap::Parser;
use parse::Token;

#[derive(Debug, Parser)]
struct Cli {
    file: String,
}

fn err_string(reason: RichReason<Token>) -> String {
    match reason {
        RichReason::ExpectedFound { expected, found } => {
            format!(
                "found {}, expected {}",
                found
                    .map(|tok| format!("'{}'", *tok))
                    .unwrap_or("end of input".to_owned()),
                if expected.is_empty() {
                    "something else".to_owned()
                } else {
                    itertools::join(expected.into_iter(), " or ")
                }
            )
        }
        RichReason::Custom(s) => s,
        RichReason::Many(errs) => itertools::join(errs.into_iter().map(err_string), "; "),
    }
}

fn main() {
    let args = Cli::parse();
    let path = &args.file;
    let input = fs::read_to_string(path).unwrap();
    match parse::parse(&input).into_result() {
        Ok(module) => println!("{}", serde_json::to_string(&module).unwrap()),
        Err(errs) => {
            for err in errs {
                Report::build(ReportKind::Error, path, err.span().start)
                    .with_message(err.to_string())
                    .with_label(
                        Label::new((path, err.span().into_range()))
                            .with_message(err_string(err.into_reason()))
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((path, Source::from(&input)))
                    .unwrap();
            }
        }
    }
}
