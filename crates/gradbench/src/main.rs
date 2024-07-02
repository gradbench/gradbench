mod lex;

use std::{fs, process::ExitCode};

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::Parser;

#[derive(Debug, Parser)]
struct Cli {
    file: String,
}

fn cli() -> Result<(), ()> {
    let args = Cli::parse();
    let path = &args.file;
    let source =
        fs::read_to_string(path).map_err(|err| eprintln!("error reading {path}: {err}"))?;
    let tokens = lex::lex(&source).map_err(|err| {
        let range = err.byte_range();
        Report::build(ReportKind::Error, path, range.start)
            .with_message("failed to tokenize")
            .with_label(
                Label::new((path, range))
                    .with_message(err.message())
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((path, Source::from(&source)))
            .unwrap()
    })?;
    for token in tokens {
        println!("{:?}", token);
    }
    Ok(())
}

fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
