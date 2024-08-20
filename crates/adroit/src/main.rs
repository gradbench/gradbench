mod compile;
mod lex;
mod lsp;
mod parse;
mod pprint;
mod range;
mod typecheck;
mod util;

use std::{fs, io, path::PathBuf, process::ExitCode, time::Instant};

use clap::{Parser, Subcommand};
use serde::Serialize;

fn read(path: PathBuf) -> Result<(PathBuf, String), Box<compile::Error>> {
    match fs::read_to_string(&path) {
        Ok(source) => Ok((path, source)),
        Err(err) => Err(Box::new(compile::Error::Read { path, err })),
    }
}

fn fmt(path: PathBuf) -> Result<(String, lex::Tokens, parse::Module), Box<compile::Error>> {
    let (path, source) = read(path)?;
    let (_, source, tokens, tree) = compile::parse(path, source)?;
    Ok((source, tokens, tree))
}

#[derive(Debug, Serialize)]
struct Graph {
    modules: compile::Modules,
    module: compile::FullModule,
}

fn entrypoint(path: PathBuf) -> Result<Graph, (compile::Modules, Box<compile::Error>)> {
    let mut modules = compile::Modules::new();
    let (path, source) = match read(path) {
        Ok(ok) => ok,
        Err(err) => return Err((modules, err)),
    };
    let (_, module) = match compile::process(&mut modules, path, source) {
        Ok(ok) => ok,
        Err(err) => return Err((modules, err)),
    };
    Ok(Graph { modules, module })
}

fn perf(n: usize, path: PathBuf) -> Result<(), (compile::Modules, Box<compile::Error>)> {
    let mut modules = compile::Modules::new();
    let (path, source) = match read(path) {
        Ok(ok) => ok,
        Err(err) => return Err((modules, err)),
    };

    let mut i = 1;
    let lexing = Instant::now();
    let tokens = loop {
        match lex::lex(&source) {
            Ok(tokens) => {
                if i < n {
                    i += 1;
                } else {
                    break tokens;
                }
            }
            Err(err) => return Err((modules, Box::new(compile::Error::Lex { path, source, err }))),
        }
    };
    let lexed = lexing.elapsed().as_secs_f64();

    let mut i = 1;
    let parsing = Instant::now();
    let tree = loop {
        match parse::parse(&tokens) {
            Ok(tree) => {
                if i < n {
                    i += 1;
                } else {
                    break tree;
                }
            }
            Err(err) => {
                return Err((
                    modules,
                    Box::new(compile::Error::Parse {
                        path,
                        source,
                        tokens,
                        err,
                    }),
                ))
            }
        }
    };
    let parsed = parsing.elapsed().as_secs_f64();

    let mut imports = vec![];
    for node in tree.imports().iter() {
        let token = node.module;
        match compile::import(&mut modules, &path, &tokens.get(token).string(&source)) {
            Ok(id) => imports.push(id),
            Err(err) => {
                return Err((
                    modules,
                    Box::new(compile::Error::Import {
                        path,
                        source,
                        tokens,
                        token,
                        err,
                    }),
                ))
            }
        }
    }

    let typing = Instant::now();
    for _ in 1..=n {
        let (module, errs) = typecheck::typecheck(
            &source,
            &tokens,
            &tree,
            imports.iter().map(|&id| &modules.get(id).module).collect(),
        );
        if !errs.is_empty() {
            let full = compile::FullModule {
                source,
                tokens,
                tree,
                imports,
                module,
            };
            return Err((
                modules,
                Box::new(compile::Error::Type {
                    path,
                    full: Box::new(full),
                    errs,
                }),
            ));
        }
    }
    let typed = typing.elapsed().as_secs_f64();

    let lines = source.lines().count();
    let total = lexed + parsed + typed;
    let integer = [lexed, parsed, typed, total]
        .iter()
        .map(|&x| x.floor().to_string().len())
        .max()
        .unwrap();
    let fractional = 3;
    let width = integer + 1 + fractional;
    let loc_per_sec = (n * lines) as f64 / total;
    println!("{n} × lex       = {lexed:>width$.fractional$} seconds",);
    println!("{n} × parse     = {parsed:>width$.fractional$} seconds",);
    println!("{n} × typecheck = {typed:>width$.fractional$} seconds",);
    println!("{n} × total     = {total:>width$.fractional$} seconds");
    println!();
    println!("{} is {lines} lines", path.display());
    println!("lines of code per second: {}", loc_per_sec.floor());

    Ok(())
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

    /// Start a language server over stdio
    Lsp,

    /// Print compiler performance info for a module
    Perf {
        /// Number of times to process the module
        #[arg(short)]
        n: usize,

        file: PathBuf,
    },
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
                compile::error(&compile::Modules::new(), *err);
                Err(())
            }
        },
        Commands::Json { file } => match entrypoint(file) {
            Ok(graph) => {
                serde_json::to_writer(io::stdout(), &graph)
                    .map_err(|err| eprintln!("error serializing module: {err}"))?;
                println!();
                Ok(())
            }
            Err((modules, err)) => {
                compile::error(&modules, *err);
                Err(())
            }
        },
        Commands::Lsp => {
            lsp::language_server();
            Ok(())
        }
        Commands::Perf { n, file } => {
            perf(n, file).map_err(|(modules, err)| compile::error(&modules, *err))
        }
    }
}

fn main() -> ExitCode {
    match cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
