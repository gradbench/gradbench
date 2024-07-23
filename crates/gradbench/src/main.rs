mod lex;
mod parse;
mod pprint;
mod range;
mod typecheck;
mod util;

use std::{fs, io, path::PathBuf, process::ExitCode};

use clap::Parser;
use indexmap::IndexMap;

fn entrypoint(
    path: PathBuf,
) -> Result<util::FullModule, (IndexMap<PathBuf, util::FullModule>, Box<util::Error>)> {
    let mut modules = IndexMap::new();
    let source = match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(err) => return Err((modules, Box::new(util::Error::Read { path, err }))),
    };
    let (_, full) = util::process(&mut modules, path, source).map_err(|err| (modules, err))?;
    Ok(full)
}

#[derive(Debug, Parser)]
struct Cli {
    file: PathBuf,
}

fn cli() -> Result<(), ()> {
    match entrypoint(Cli::parse().file) {
        Ok(full) => {
            serde_json::to_writer(io::stdout(), &full.module)
                .map_err(|err| eprintln!("error serializing module: {err}"))?;
            println!();
            Ok(())
        }
        Err((modules, err)) => {
            util::error(&modules, *err);
            Err(())
        }
    }
}

fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
