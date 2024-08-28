mod cli;
mod compile;
mod fetch;
mod graph;
mod lex;
mod lsp;
mod parse;
mod pprint;
mod range;
mod typecheck;
mod util;

use std::process::ExitCode;

fn main() -> ExitCode {
    match cli::cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
