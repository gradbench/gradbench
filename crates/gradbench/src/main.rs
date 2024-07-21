mod lex;
mod parse;
mod pprint;
mod range;
mod typecheck;
mod util;

use std::{fs, io, process::ExitCode};

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
    let autodiff = typecheck::autodiff();
    let math = typecheck::math();
    let module = typecheck::typecheck(
        |name| match name {
            "array" => &array,
            "autodiff" => &autodiff,
            "math" => &math,
            _ => panic!("unknown module: {name}"),
        },
        &source,
        &tokens,
        &tree,
    )
    .map_err(|(module, err)| {
        let printer = util::Printer {
            source: &source,
            tokens: &tokens,
            module: &module,
        };
        let (range, message) = match err {
            typecheck::TypeError::TooManyImports => todo!(),
            typecheck::TypeError::TooManyTypes => todo!(),
            typecheck::TypeError::TooManyFields => todo!(),
            typecheck::TypeError::Undefined { name } => {
                (tokens.get(name).byte_range(), "undefined".to_owned())
            }
            typecheck::TypeError::Duplicate { name } => {
                (tokens.get(name).byte_range(), "duplicate".to_owned())
            }
            typecheck::TypeError::Untyped { name } => {
                (tokens.get(name).byte_range(), "untyped".to_owned())
            }
            typecheck::TypeError::Param {
                id,
                expected,
                actual,
            } => (
                range::param_range(&tokens, &tree, id),
                format!(
                    "expected `{}`, got `{}`",
                    printer.ty(expected),
                    printer.ty(actual)
                ),
            ),
            typecheck::TypeError::Expr {
                id,
                expected,
                actual,
            } => (
                range::expr_range(&tokens, &tree, id),
                format!(
                    "expected `{}`, got `{}`",
                    printer.ty(expected),
                    printer.ty(actual)
                ),
            ),
            typecheck::TypeError::NotPair { param, ty } => (
                range::param_range(&tokens, &tree, param),
                format!("expected tuple, got `{}`", printer.ty(ty)),
            ),
            typecheck::TypeError::NotFunction { expr, ty } => (
                range::expr_range(&tokens, &tree, expr),
                format!("expected function, got `{}`", printer.ty(ty)),
            ),
        };
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
    serde_json::to_writer(io::stdout(), &module)
        .map_err(|err| eprintln!("error serializing module: {err}"))?;
    println!();
    Ok(())
}

fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
