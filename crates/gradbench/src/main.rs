mod json;
mod lex;
mod parse;
mod pprint;
mod typecheck;
mod util;

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
            .unwrap();
    })?;
    let tree = parse::parse(&tokens).map_err(|err| {
        let (id, message) = match err {
            parse::ParseError::Expected { id, kinds } => (
                id,
                format!(
                    "expected {}",
                    itertools::join(kinds.into_iter().map(|kind| kind.to_string()), " or ")
                ),
            ),
        };
        let range = tokens.get(id).byte_range();
        Report::build(ReportKind::Error, path, range.start)
            .with_message("failed to parse")
            .with_label(
                Label::new((path, range))
                    .with_message(message)
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((path, Source::from(&source)))
            .unwrap();
    })?;
    let array = typecheck::array();
    let math = typecheck::math();
    let module = typecheck::typecheck(
        |name| match name {
            "array" => &array,
            "math" => &math,
            _ => panic!(),
        },
        &source,
        &tokens,
        &tree,
    )
    .map_err(|err| {
        let (id, message) = match err {
            typecheck::TypeError::TooManyImports => todo!(),
            typecheck::TypeError::TooManyTypes => todo!(),
            typecheck::TypeError::TooManyFields => todo!(),
            typecheck::TypeError::Undefined { name } => (name, "undefined"),
            typecheck::TypeError::Duplicate { name } => (name, "duplicate"),
            typecheck::TypeError::Untyped { name } => (name, "untyped"),
        };
        let range = tokens.get(id).byte_range();
        Report::build(ReportKind::Error, path, range.start)
            .with_message("failed to typecheck")
            .with_label(
                Label::new((path, range))
                    .with_message(message)
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((path, Source::from(&source)))
            .unwrap();
    })?;
    println!("{module:?}");
    Ok(())
}

fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
