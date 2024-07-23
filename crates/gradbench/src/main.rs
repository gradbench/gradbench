mod lex;
mod parse;
mod pprint;
mod range;
mod typecheck;
mod util;

use std::{
    fs, io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::Parser;
use indexmap::IndexMap;

fn builtin(name: &str) -> Option<&'static str> {
    match name {
        "array" => Some(include_str!("modules/array.adroit")),
        "autodiff" => Some(include_str!("modules/autodiff.adroit")),
        "math" => Some(include_str!("modules/math.adroit")),
        _ => None,
    }
}

fn resolve(from: &Path, name: &str) -> io::Result<PathBuf> {
    if builtin(name).is_some() {
        let mut path = dirs::cache_dir()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no cache directory"))?;
        path.push("adroit/modules");
        path.push(name);
        assert!(path.set_extension("adroit"));
        Ok(path)
    } else {
        let dir = from
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no parent directory"))?;
        dir.join(name).canonicalize()
    }
}

fn read(name: &str, path: &Path) -> io::Result<String> {
    if let Some(source) = builtin(name) {
        fs::create_dir_all(path.parent().expect("join should imply parent"))?;
        fs::write(path, source)?;
        Ok(source.to_owned())
    } else {
        fs::read_to_string(path)
    }
}

#[derive(Debug)]
struct FullModule {
    source: String,
    tokens: lex::Tokens,
    tree: parse::Module,
    imports: Vec<usize>,
    module: typecheck::Module,
}

#[derive(Debug)]
enum Error {
    Resolve {
        err: io::Error,
    },
    Read {
        path: PathBuf,
        err: io::Error,
    },
    Lex {
        path: PathBuf,
        source: String,
        err: lex::LexError,
    },
    Parse {
        path: PathBuf,
        source: String,
        tokens: lex::Tokens,
        err: parse::ParseError,
    },
    Import {
        path: PathBuf,
        source: String,
        tokens: lex::Tokens,
        token: lex::TokenId,
        err: Box<Error>,
    },
    Type {
        path: PathBuf,
        source: String,
        tokens: lex::Tokens,
        tree: parse::Module,
        module: Box<typecheck::Module>,
        err: typecheck::TypeError,
    },
}

fn import(
    modules: &mut IndexMap<PathBuf, FullModule>,
    from: &Path,
    name: &str,
) -> Result<usize, Box<Error>> {
    let path = resolve(from, name).map_err(|err| Error::Resolve { err })?;
    if let Some(i) = modules.get_index_of(&path) {
        return Ok(i);
    }
    let source = match read(name, &path) {
        Ok(source) => source,
        Err(err) => return Err(Box::new(Error::Read { path, err })),
    };
    let (path, full) = process(modules, path, source)?;
    let (i, prev) = modules.insert_full(path, full);
    assert!(prev.is_none(), "cyclic import should yield stack overflow");
    Ok(i)
}

fn process(
    modules: &mut IndexMap<PathBuf, FullModule>,
    path: PathBuf,
    source: String,
) -> Result<(PathBuf, FullModule), Box<Error>> {
    let tokens = match lex::lex(&source) {
        Ok(tokens) => tokens,
        Err(err) => {
            return Err(Box::new(Error::Lex {
                path,
                source: source.to_owned(),
                err,
            }))
        }
    };
    let tree = match parse::parse(&tokens) {
        Ok(tree) => tree,
        Err(err) => {
            return Err(Box::new(Error::Parse {
                path,
                source: source.to_owned(),
                tokens,
                err,
            }))
        }
    };
    let mut imports = vec![];
    for node in tree.imports().iter() {
        let token = node.module;
        match import(modules, &path, &tokens.get(token).string(&source)) {
            Ok(i) => imports.push(i),
            Err(err) => {
                return Err(Box::new(Error::Import {
                    path,
                    source: source.to_owned(),
                    tokens,
                    token,
                    err,
                }))
            }
        }
    }
    let module = match typecheck::typecheck(
        &source,
        &tokens,
        &tree,
        imports.iter().map(|&i| &modules[i].module).collect(),
    ) {
        Ok(module) => module,
        Err((module, err)) => {
            return Err(Box::new(Error::Type {
                path,
                source: source.to_owned(),
                tokens,
                tree,
                module,
                err,
            }))
        }
    };
    let full = FullModule {
        source,
        tokens,
        tree,
        imports,
        module,
    };
    Ok((path, full))
}

fn error(err: Error) {
    match err {
        Error::Resolve { err } => eprintln!("error resolving module: {err}"),
        Error::Read { path, err } => eprintln!("error reading {}: {err}", path.display()),
        Error::Lex { path, source, err } => {
            let path = &path.display().to_string();
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
        }
        Error::Parse {
            path,
            source,
            tokens,
            err,
        } => {
            let path = &path.display().to_string();
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
        }
        Error::Import {
            path,
            source,
            tokens,
            token,
            err,
        } => {
            error(*err);
            let path = &path.display().to_string();
            let range = tokens.get(token).byte_range();
            Report::build(ReportKind::Error, path, range.start)
                .with_message("failed to import")
                .with_label(Label::new((path, range)).with_color(Color::Red))
                .finish()
                .eprint((path, Source::from(&source)))
                .unwrap();
        }
        Error::Type {
            path,
            source,
            tokens,
            tree,
            module,
            err,
        } => {
            let path = &path.display().to_string();
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
        }
    }
}

fn entrypoint(path: PathBuf) -> Result<FullModule, Box<Error>> {
    let source = match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(err) => return Err(Box::new(Error::Read { path, err })),
    };
    let mut modules = IndexMap::new();
    let (_, full) = process(&mut modules, path, source)?;
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
        Err(err) => {
            error(*err);
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
