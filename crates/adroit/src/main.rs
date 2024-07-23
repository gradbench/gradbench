mod lex;
mod parse;
mod pprint;
mod range;
mod typecheck;
mod util;

use std::{fs, io, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use indexmap::IndexMap;

fn read(path: PathBuf) -> Result<(PathBuf, String), Box<util::Error>> {
    match fs::read_to_string(&path) {
        Ok(source) => Ok((path, source)),
        Err(err) => Err(Box::new(util::Error::Read { path, err })),
    }
}

fn fmt(path: PathBuf) -> Result<(String, lex::Tokens, parse::Module), Box<util::Error>> {
    let (path, source) = read(path)?;
    let (_, source, tokens, tree) = util::parse(path, source)?;
    Ok((source, tokens, tree))
}

fn entrypoint(
    path: PathBuf,
) -> Result<util::FullModule, (IndexMap<PathBuf, util::FullModule>, Box<util::Error>)> {
    let mut modules = IndexMap::new();
    let (path, source) = match read(path) {
        Ok(ok) => ok,
        Err(err) => return Err((modules, err)),
    };
    let (_, full) = util::process(&mut modules, path, source).map_err(|err| (modules, err))?;
    Ok(full)
}

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Print the reformatted source code of a module
    Fmt { file: PathBuf },

    /// Print the typed IR of a module as JSON
    Json { file: PathBuf },
}

fn cli() -> Result<(), ()> {
    match Cli::parse().command {
        Commands::Fmt { file } => match fmt(file) {
            Ok((source, tokens, module)) => {
                pprint::pprint(&mut io::stdout(), &source, &tokens, &module)
                    .map_err(|err| eprintln!("error formatting module: {err}"))?;
                Ok(())
            }
            Err(err) => {
                util::error(&IndexMap::new(), *err);
                Err(())
            }
        },
        Commands::Json { file } => match entrypoint(file) {
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
        },
    }
}

fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
