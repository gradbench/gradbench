mod parse;

use std::fs;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::Parser;

#[derive(Debug, Parser)]
struct Cli {
    file: String,
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
                            .with_message(err.reason().to_string())
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((path, Source::from(&input)))
                    .unwrap();
            }
        }
    }
}
