use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

use ariadne::{Color, Label, Report, ReportKind, Source};
use disjoint_sets::ElementType;
use indexmap::IndexMap;
use serde::{ser::SerializeSeq, Serialize, Serializer};

use crate::{lex, parse, range, typecheck};

pub fn u32_to_usize(n: u32) -> usize {
    n.try_into()
        .expect("pointer size is assumed to be at least 32 bits")
}

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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ModuleId {
    pub index: usize, // just for convenience for now; can definitely choose a smaller type
}

impl ElementType for ModuleId {
    fn from_usize(n: usize) -> Option<Self> {
        Some(Self { index: n })
    }

    fn to_usize(self) -> usize {
        self.index
    }
}

#[derive(Debug, Serialize)]
pub struct FullModule {
    pub source: String,
    pub tokens: lex::Tokens,
    pub tree: parse::Module,
    pub imports: Vec<ModuleId>,
    pub module: typecheck::Module,
}

#[derive(Debug)]
pub struct Modules {
    modules: IndexMap<PathBuf, FullModule>,
}

impl Modules {
    pub fn new() -> Self {
        Self {
            modules: IndexMap::new(),
        }
    }

    pub fn get(&self, id: ModuleId) -> &FullModule {
        &self.modules[id.to_usize()]
    }
}

impl Serialize for Modules {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.modules.len()))?;
        for full in self.modules.values() {
            seq.serialize_element(full)?;
        }
        seq.end()
    }
}

#[derive(Debug)]
pub enum Error {
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
        full: Box<FullModule>,
        err: typecheck::TypeError,
    },
}

pub fn parse(
    path: PathBuf,
    source: String,
) -> Result<(PathBuf, String, lex::Tokens, parse::Module), Box<Error>> {
    let tokens = match lex::lex(&source) {
        Ok(tokens) => tokens,
        Err(err) => return Err(Box::new(Error::Lex { path, source, err })),
    };
    let tree = match parse::parse(&tokens) {
        Ok(tree) => tree,
        Err(err) => {
            return Err(Box::new(Error::Parse {
                path,
                source,
                tokens,
                err,
            }))
        }
    };
    Ok((path, source, tokens, tree))
}

pub fn process(
    modules: &mut Modules,
    path: PathBuf,
    source: String,
) -> Result<(PathBuf, FullModule), Box<Error>> {
    let (path, source, tokens, tree) = parse(path, source)?;
    let mut imports = vec![];
    for node in tree.imports().iter() {
        let token = node.module;
        match import(modules, &path, &tokens.get(token).string(&source)) {
            Ok(id) => imports.push(id),
            Err(err) => {
                return Err(Box::new(Error::Import {
                    path,
                    source,
                    tokens,
                    token,
                    err,
                }))
            }
        }
    }
    let (module, res) = typecheck::typecheck(
        &source,
        &tokens,
        &tree,
        imports.iter().map(|&id| &modules.get(id).module).collect(),
    );
    let full = FullModule {
        source,
        tokens,
        tree,
        imports,
        module,
    };
    match res {
        Ok(()) => Ok((path, full)),
        Err(err) => Err(Box::new(Error::Type {
            path,
            full: Box::new(full),
            err,
        })),
    }
}

pub fn import(modules: &mut Modules, from: &Path, name: &str) -> Result<ModuleId, Box<Error>> {
    let path = resolve(from, name).map_err(|err| Error::Resolve { err })?;
    if let Some(index) = modules.modules.get_index_of(&path) {
        return Ok(ModuleId { index });
    }
    let source = match read(name, &path) {
        Ok(source) => source,
        Err(err) => return Err(Box::new(Error::Read { path, err })),
    };
    let (path, full) = process(modules, path, source)?;
    let (index, prev) = modules.modules.insert_full(path, full);
    assert!(prev.is_none(), "cyclic import should yield stack overflow");
    Ok(ModuleId { index })
}

pub fn error(modules: &Modules, err: Error) {
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
            error(modules, *err);
            let path = &path.display().to_string();
            let range = tokens.get(token).byte_range();
            Report::build(ReportKind::Error, path, range.start)
                .with_message("failed to import")
                .with_label(Label::new((path, range)).with_color(Color::Red))
                .finish()
                .eprint((path, Source::from(&source)))
                .unwrap();
        }
        Error::Type { path, full, err } => {
            let path = &path.display().to_string();
            let FullModule {
                source,
                tokens,
                tree,
                imports: _,
                module,
            } = &*full;
            let printer = Printer {
                modules,
                full: &full,
            };
            let (range, message) = match err {
                typecheck::TypeError::TooManyImports => todo!("too many imports"),
                typecheck::TypeError::TooManyTypes => todo!("too many types"),
                typecheck::TypeError::TooManyFields => todo!("too many fields"),
                typecheck::TypeError::Undefined { name } => {
                    (tokens.get(name).byte_range(), "undefined".to_owned())
                }
                typecheck::TypeError::Duplicate { name } => {
                    (tokens.get(name).byte_range(), "duplicate".to_owned())
                }
                typecheck::TypeError::Untyped { name } => {
                    (tokens.get(name).byte_range(), "untyped".to_owned())
                }
                typecheck::TypeError::Type {
                    id,
                    expected,
                    actual,
                } => (
                    range::ty_range(tokens, tree, id),
                    format!(
                        "expected `{}`, got `{}`",
                        printer.ty(expected),
                        printer.ty(actual)
                    ),
                ),
                typecheck::TypeError::Param {
                    id,
                    expected,
                    actual,
                } => (
                    range::param_range(tokens, tree, id),
                    format!(
                        "expected `{}`, got `{}`",
                        printer.ty(expected),
                        printer.ty(actual)
                    ),
                ),
                typecheck::TypeError::Expr { id, expected } => {
                    let actual = module.val(module.expr(id)).ty;
                    (
                        range::expr_range(tokens, tree, id),
                        format!(
                            "expected `{}`, got `{}`",
                            printer.ty(expected),
                            printer.ty(actual)
                        ),
                    )
                }
                typecheck::TypeError::NotPoly { expr } => {
                    let ty = module.val(module.expr(expr)).ty;
                    (
                        range::expr_range(tokens, tree, expr),
                        format!("expected polymorphic type, got `{}`", printer.ty(ty)),
                    )
                }
                typecheck::TypeError::NotNumber { expr } => {
                    let ty = module.val(module.expr(expr)).ty;
                    (
                        range::expr_range(tokens, tree, expr),
                        format!("expected number, got `{}`", printer.ty(ty)),
                    )
                }
                typecheck::TypeError::NotVector { expr } => {
                    let ty = module.val(module.expr(expr)).ty;
                    (
                        range::expr_range(tokens, tree, expr),
                        format!("expected number or vector, got `{}`", printer.ty(ty)),
                    )
                }
                typecheck::TypeError::NotPair { param } => {
                    let ty = module.val(module.param(param)).ty;
                    (
                        range::param_range(tokens, tree, param),
                        format!("expected tuple, got `{}`", printer.ty(ty)),
                    )
                }
                typecheck::TypeError::NotArray { expr } => {
                    let ty = module.val(module.expr(expr)).ty;
                    (
                        range::expr_range(tokens, tree, expr),
                        format!("expected array, got `{}`", printer.ty(ty)),
                    )
                }
                typecheck::TypeError::NotFunc { expr } => {
                    let ty = module.val(module.expr(expr)).ty;
                    (
                        range::expr_range(tokens, tree, expr),
                        format!("expected function, got `{}`", printer.ty(ty)),
                    )
                }
                typecheck::TypeError::WrongRecord { param, ty } => (
                    range::param_range(tokens, tree, param),
                    format!("expected `{}`", printer.ty(ty)),
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

#[derive(Clone, Copy, Debug)]
struct Printer<'a> {
    modules: &'a Modules,
    full: &'a FullModule,
}

impl Printer<'_> {
    fn ty(&self, id: typecheck::TypeId) -> Type {
        Type { printer: *self, id }
    }

    fn get_ty(&self, id: typecheck::TypeId) -> typecheck::Type {
        self.full.module.ty(id)
    }

    fn print_ty(&self, w: &mut impl fmt::Write, id: typecheck::TypeId) -> fmt::Result {
        use typecheck::Type::*;
        match self.get_ty(id) {
            Untyped => write!(w, "?")?,
            Var { src, def } => {
                let full = match src {
                    Some(id) => self.modules.get(self.full.imports[id.to_usize()]),
                    None => self.full,
                };
                write!(w, "{}", &full.source[full.tokens.get(def).byte_range()])?;
            }
            Poly { var, inner } => {
                write!(w, "{} => {}", self.ty(var), self.ty(inner))?;
            }
            Unit => write!(w, "()")?,
            Int => write!(w, "Int")?,
            Float => write!(w, "Float")?,
            Prod { fst, snd } => {
                if let Prod { .. } | Sum { .. } | Func { .. } = self.get_ty(fst) {
                    write!(w, "({})", self.ty(fst))
                } else {
                    write!(w, "{}", self.ty(fst))
                }?;
                write!(w, " * ")?;
                if let Sum { .. } | Func { .. } = self.get_ty(snd) {
                    write!(w, "({})", self.ty(snd))
                } else {
                    write!(w, "{}", self.ty(snd))
                }?;
            }
            Sum { left, right } => {
                if let Prod { .. } | Sum { .. } | Func { .. } = self.get_ty(left) {
                    write!(w, "({})", self.ty(left))
                } else {
                    write!(w, "{}", self.ty(left))
                }?;
                write!(w, " + ")?;
                if let Func { .. } = self.get_ty(right) {
                    write!(w, "({})", self.ty(right))
                } else {
                    write!(w, "{}", self.ty(right))
                }?;
            }
            Array { index, elem } => {
                write!(w, "[{}]", self.ty(index))?;
                if let Prod { .. } | Sum { .. } | Func { .. } = self.get_ty(elem) {
                    write!(w, "({})", self.ty(elem))
                } else {
                    write!(w, "{}", self.ty(elem))
                }?;
            }
            Record { name, field, rest } => {
                write!(w, "{{")?;
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    write!(w, "{}: {}", self.full.module.field(n), self.ty(v))?;
                    match self.get_ty(r) {
                        Record { name, field, rest } => {
                            write!(w, ", ")?;
                            (n, v, r) = (name, field, rest);
                        }
                        End => break,
                        _ => panic!("invalid record"),
                    }
                }
                write!(w, "}}")?;
            }
            End => write!(w, "{{}}")?,
            Func { dom, cod } => {
                if let Func { .. } = self.get_ty(dom) {
                    write!(w, "({})", self.ty(dom))
                } else {
                    write!(w, "{}", self.ty(dom))
                }?;
                write!(w, " -> {}", self.ty(cod))?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Type<'a> {
    printer: Printer<'a>,
    id: typecheck::TypeId,
}

impl fmt::Display for Type<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.printer.print_ty(f, self.id)
    }
}
