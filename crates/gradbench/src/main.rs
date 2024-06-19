mod parse;

use std::{fs, process};

use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::error::RichReason;
use clap::Parser;
use parse::Token;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long, value_name = "FILENAME", default_value = "gradbench.adroit")]
    defs: String,

    #[arg(long, value_name = "FILENAME", default_value = "gradbench.json")]
    config: String,
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

    let mut config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&args.config).unwrap()).unwrap();

    let path = &args.defs;
    let input = fs::read_to_string(path).unwrap();
    let module = match parse::parse(&input).into_result() {
        Ok(module) => module,
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
            process::exit(1);
        }
    };

    config.as_object_mut().unwrap().insert(
        "defs".to_owned(),
        serde_json::to_value(module.defs).unwrap(),
    );
    println!("{}", serde_json::to_string(&config).unwrap());
}
