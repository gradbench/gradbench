mod lex;
mod parse;
mod pprint;
mod util;

use std::{fmt, fs, process::ExitCode};

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::Parser;

struct ModuleWithSource {
    source: String,
    tokens: lex::Tokens,
    module: parse::Module,
}

impl fmt::Display for ModuleWithSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        pprint::pprint(f, &self.source, &self.tokens, &self.module)
    }
}

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
    let module = parse::parse(&tokens).map_err(|err| {
        use lex::TokenKind::*;
        use parse::ParseError::*;
        let (id, message) = match err {
            Expected { id, kind } => (id, format!("expected {}", kind)),
            ExpectedType { id } => (id, format!("expected {} or {}", Ident, LParen)),
            ExpectedBind { id } => (id, format!("expected {} or {}", Ident, LParen)),
            BindPairRightMissing { id } => (id, format!("expected {}", Comma)),
            ExpectedExpression { id } => (id, "expected expression".to_owned()),
            UnexpectedToplevel { id } => (id, format!("expected {} or {}", Def, Eof)),
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
    print!(
        "{}",
        ModuleWithSource {
            source,
            tokens,
            module
        }
    );
    Ok(())
}

fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
