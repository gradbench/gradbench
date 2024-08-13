use std::{
    fmt::{self, Write},
    fs, io,
    marker::PhantomData,
    ops::Range,
    path::{Path, PathBuf},
};

use ariadne::{Cache, Color, Label, Report, ReportBuilder, ReportKind, Source};
use indexmap::IndexMap;
use serde::{ser::SerializeSeq, Serialize, Serializer};

use crate::{
    lex, parse,
    range::{bind_range, expr_range, param_range, ty_range},
    typecheck,
    util::{Diagnostic, Emitter, Id},
};

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

impl Id for ModuleId {
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
        errs: Vec<typecheck::TypeError>,
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
    let (module, errs) = typecheck::typecheck(
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
    if errs.is_empty() {
        Ok((path, full))
    } else {
        Err(Box::new(Error::Type {
            path,
            full: Box::new(full),
            errs,
        }))
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

#[derive(Debug)]
struct AriadneEmitter<'a, C: Cache<&'a str>> {
    cache: C,
    message: String,
    phantom: PhantomData<&'a ()>,
}

impl<'a, C: Cache<&'a str>> AriadneEmitter<'a, C> {
    fn new(cache: C, message: impl ToString) -> Self {
        Self {
            cache,
            message: message.to_string(),
            phantom: PhantomData,
        }
    }
}

impl<'a, C: Cache<&'a str>> Emitter<(&'a str, Range<usize>)> for AriadneEmitter<'a, C> {
    fn diagnostic(
        &mut self,
        span: (&'a str, Range<usize>),
        message: impl ToString,
    ) -> impl Diagnostic<(&'a str, std::ops::Range<usize>)> {
        let (path, range) = span.clone();
        AriadneDiagnostic {
            cache: &mut self.cache,
            builder: Report::build(ReportKind::Error, path, range.start)
                .with_message(&self.message)
                .with_label(
                    Label::new(span)
                        .with_color(Color::Red)
                        .with_message(message),
                ),
        }
    }
}

#[derive(Debug)]
struct AriadneDiagnostic<'a, 'b, C: Cache<&'a str>> {
    cache: &'b mut C,
    builder: ReportBuilder<'a, (&'a str, Range<usize>)>,
}

impl<'a, 'b, C: Cache<&'a str>> Diagnostic<(&'a str, Range<usize>)>
    for AriadneDiagnostic<'a, 'b, C>
{
    fn related(mut self, span: (&'a str, Range<usize>), message: impl ToString) -> Self {
        self.builder.add_label(
            Label::new(span)
                .with_color(Color::Blue)
                .with_message(message),
        );
        self
    }

    fn finish(self) {
        self.builder.finish().eprint(self.cache).unwrap();
    }
}

pub fn error(modules: &Modules, err: Error) {
    match err {
        Error::Resolve { err } => eprintln!("error resolving module: {err}"),
        Error::Read { path, err } => eprintln!("error reading {}: {err}", path.display()),
        Error::Lex { path, source, err } => {
            let path: &str = &path.display().to_string();
            AriadneEmitter::new((path, Source::from(&source)), "failed to tokenize")
                .diagnostic((path, err.byte_range()), err.message())
                .finish();
        }
        Error::Parse {
            path,
            source,
            tokens,
            err,
        } => {
            let path: &str = &path.display().to_string();
            let (id, message) = match err {
                parse::ParseError::Expected { id, kinds } => (
                    id,
                    format!(
                        "expected {}",
                        itertools::join(kinds.into_iter().map(|kind| kind.to_string()), " or ")
                    ),
                ),
            };
            AriadneEmitter::new((path, Source::from(&source)), "failed to parse")
                .diagnostic((path, tokens.get(id).byte_range()), message)
                .finish();
        }
        Error::Import {
            path,
            source,
            tokens,
            token,
            err,
        } => {
            error(modules, *err);
            let path: &str = &path.display().to_string();
            AriadneEmitter::new((path, Source::from(&source)), "failed to import")
                .diagnostic((path, tokens.get(token).byte_range()), "this one")
                .finish();
        }
        Error::Type { path, full, errs } => {
            let path: &str = &path.display().to_string();
            let printer = Printer::new(modules, &full);
            let mut emitter =
                AriadneEmitter::new((path, Source::from(&full.source)), "failed to typecheck");
            for err in errs {
                printer.emit_type_error(&mut emitter, path, err);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Printer<'a> {
    modules: &'a Modules,
    full: &'a FullModule,
}

impl<'a> Printer<'a> {
    pub fn new(modules: &'a Modules, full: &'a FullModule) -> Self {
        Self { modules, full }
    }
}

impl<'a> Printer<'a> {
    fn get_ty(&self, id: typecheck::TypeId) -> typecheck::Type {
        self.full.module.ty(id)
    }

    fn token_range(&self, id: lex::TokenId) -> Range<usize> {
        self.full.tokens.get(id).byte_range()
    }

    fn ty_range(&self, id: parse::TypeId) -> Range<usize> {
        ty_range(&self.full.tokens, &self.full.tree, id)
    }

    fn bind_range(&self, id: parse::ParamId) -> Range<usize> {
        bind_range(&self.full.tokens, &self.full.tree, id)
    }

    fn param_range(&self, id: parse::ParamId) -> Range<usize> {
        param_range(&self.full.tokens, &self.full.tree, id)
    }

    fn expr_range(&self, id: parse::ExprId) -> Range<usize> {
        expr_range(&self.full.tokens, &self.full.tree, id)
    }

    fn ty(&self, id: typecheck::TypeId) -> Type {
        Type { printer: *self, id }
    }

    fn param_ty(&self, id: parse::ParamId) -> Type {
        self.ty(self.full.module.val(self.full.module.param(id)).ty)
    }

    fn expr_ty(&self, id: parse::ExprId) -> Type {
        self.ty(self.full.module.val(self.full.module.expr(id)).ty)
    }

    fn print_ty(&self, w: &mut impl fmt::Write, id: typecheck::TypeId) -> fmt::Result {
        use typecheck::Type::*;
        match self.get_ty(id) {
            Unknown { id: _ } => write!(w, "_")?,
            Scalar { id: _ } => write!(w, "_")?,
            Vector { id: _, scalar: _ } => write!(w, "_")?,
            Fragment => panic!("fragment type should not be printed"),
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

    pub fn emit_type_error(
        &self,
        emitter: &mut impl Emitter<(&'a str, Range<usize>)>,
        path: &'a str,
        err: typecheck::TypeError,
    ) {
        use typecheck::TypeError::*;
        match err {
            TooManyImports => todo!("too many imports"),
            TooManyTypes => todo!("too many types"),
            TooManyFields => todo!("too many fields"),
            Undefined { name } => emitter
                .diagnostic((path, self.token_range(name)), "undefined")
                .finish(),
            Duplicate { name } => emitter
                .diagnostic((path, self.token_range(name)), "duplicate")
                .finish(),
            Dom { name } | Cod { name } => emitter
                .diagnostic((path, self.token_range(name)), "untyped")
                .finish(),
            Param { id } => emitter
                .diagnostic(
                    (path, self.bind_range(id)),
                    format!("inferred type: `{}`", self.param_ty(id)),
                )
                .related(
                    (path, self.ty_range(self.full.tree.param(id).ty.unwrap())),
                    "does not match the given type",
                )
                .finish(),
            Elem { id } => match self.full.tree.expr(id) {
                parse::Expr::Elem { array, index } => emitter
                    .diagnostic(
                        (path, self.expr_range(index)),
                        format!("index type does not match: `{}`", self.expr_ty(index)),
                    )
                    .related(
                        (path, self.expr_range(array)),
                        format!("array type: `{}`", self.expr_ty(array)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            Inst { mut id } => {
                let range = self.expr_range(id);
                let mut m = 0;
                while let parse::Expr::Inst { val, ty: _ } = self.full.tree.expr(id) {
                    m += 1;
                    id = val;
                }
                let mut v = self.full.module.expr(id);
                while let typecheck::Src::Inst { val, ty: _ } = self.full.module.val(v).src {
                    v = val;
                }
                match self.full.module.val(v).src {
                    typecheck::Src::Def { id } => {
                        let def = self.full.tree.def(id);
                        let n = def.types.len();
                        emitter
                            .diagnostic(
                                (path, range),
                                format!("{m} type arguments is {} too many", m - n),
                            )
                            .related(
                                (path, self.token_range(def.name)),
                                format!("function only takes {n} type parameters"),
                            )
                            .finish()
                    }
                    _ => todo!("too many type arguments for imported function"),
                }
            }
            Apply { id } => match self.full.tree.expr(id) {
                parse::Expr::Apply { func, arg } => emitter
                    .diagnostic(
                        (path, self.expr_range(arg)),
                        format!("argument type does not match: `{}`", self.expr_ty(arg)),
                    )
                    .related(
                        (path, self.expr_range(func)),
                        format!("function type: `{}`", self.expr_ty(func)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            MapLhs { id } => match self.full.tree.expr(id) {
                parse::Expr::Map { func, arg: _ } => emitter
                    .diagnostic(
                        (path, self.expr_range(func)),
                        format!("expected a function but instead: `{}`", self.expr_ty(func)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            MapRhs { id } => match self.full.tree.expr(id) {
                parse::Expr::Map { func, arg } => emitter
                    .diagnostic(
                        (path, self.expr_range(arg)),
                        format!(
                            "argument type is not a matching array: `{}`",
                            self.expr_ty(arg)
                        ),
                    )
                    .related(
                        (path, self.expr_range(func)),
                        format!(
                            "function type to be mapped over an array: `{}`",
                            self.expr_ty(func)
                        ),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            Let { id } => match self.full.tree.expr(id) {
                parse::Expr::Let {
                    param,
                    val,
                    body: _,
                } => emitter
                    .diagnostic(
                        (path, self.expr_range(val)),
                        format!("inferred type: `{}`", self.expr_ty(val)),
                    )
                    .related(
                        (path, self.param_range(param)),
                        format!(
                            "expected to match the binding type: `{}`",
                            self.param_ty(param)
                        ),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            Index { id } => match self.full.tree.expr(id) {
                parse::Expr::Index {
                    name: _,
                    val,
                    body: _,
                } => emitter
                    .diagnostic(
                        (path, self.expr_range(val)),
                        format!("expected `Int` but instead: `{}`", self.expr_ty(val)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            Neg { id } => match self.full.tree.expr(id) {
                parse::Expr::Unary { op: _, arg } => emitter
                    .diagnostic(
                        (path, self.expr_range(arg)),
                        format!("not a scalar or vector: `{}`", self.expr_ty(arg)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            ElemLhs { id } | DivLhs { id } => match self.full.tree.expr(id) {
                parse::Expr::Binary { lhs, op: _, rhs: _ } => emitter
                    .diagnostic(
                        (path, self.expr_range(lhs)),
                        format!("not a scalar or vector: `{}`", self.expr_ty(lhs)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            ElemRhs { id } => match self.full.tree.expr(id) {
                parse::Expr::Binary { lhs, op: _, rhs } => emitter
                    .diagnostic(
                        (path, self.expr_range(rhs)),
                        format!("right-hand type: `{}`", self.expr_ty(rhs)),
                    )
                    .related(
                        (path, self.expr_range(lhs)),
                        format!(
                            "does not match left-hand scalar or vector: `{}`",
                            self.expr_ty(lhs)
                        ),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            MulLhs { id } => match self.full.tree.expr(id) {
                parse::Expr::Binary { lhs, op: _, rhs: _ } => emitter
                    .diagnostic(
                        (path, self.expr_range(lhs)),
                        format!("not a scalar: `{}`", self.expr_ty(lhs)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            MulRhs { id } => match self.full.tree.expr(id) {
                parse::Expr::Binary { lhs, op: _, rhs } => emitter
                    .diagnostic(
                        (path, self.expr_range(rhs)),
                        format!("not a matching scalar or vector: `{}`", self.expr_ty(rhs)),
                    )
                    .related(
                        (path, self.expr_range(lhs)),
                        format!("left-hand scalar: `{}`", self.expr_ty(lhs)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            DivRhs { id } => match self.full.tree.expr(id) {
                parse::Expr::Binary { lhs, op: _, rhs } => emitter
                    .diagnostic(
                        (path, self.expr_range(rhs)),
                        format!("not a matching scalar: `{}`", self.expr_ty(rhs)),
                    )
                    .related(
                        (path, self.expr_range(lhs)),
                        format!("left-hand scalar or vector: `{}`", self.expr_ty(lhs)),
                    )
                    .finish(),
                _ => unreachable!(),
            },
            Lambda { id } => match self.full.tree.expr(id) {
                parse::Expr::Lambda { param: _, ty, body } => emitter
                    .diagnostic(
                        (path, self.expr_range(body)),
                        format!("inferred type: `{}`", self.expr_ty(body),),
                    )
                    .related(
                        (path, self.ty_range(ty.unwrap())),
                        "does not match the given type",
                    )
                    .finish(),
                _ => unreachable!(),
            },
            Def { id } => {
                let &parse::Def { ty, body, .. } = self.full.tree.def(id);
                emitter
                    .diagnostic(
                        (path, self.expr_range(body)),
                        format!("inferred type: `{}`", self.expr_ty(body)),
                    )
                    .related(
                        (path, self.ty_range(ty.unwrap())),
                        "does not match the given type",
                    )
                    .finish()
            }
            AmbigParam { id } => emitter
                .diagnostic(
                    (path, self.param_range(id)),
                    format!("ambiguous type: `{}`", self.param_ty(id)),
                )
                .finish(),
            AmbigTypeArgs { id } => {
                let range = self.expr_range(id);
                let mut message = String::new();
                write!(message, "ambiguous type arguments: ").unwrap();
                write!(message, "`{}[", &self.full.source[range.clone()]).unwrap();
                let mut args = vec![];
                let mut v = self.full.module.expr(id);
                while let typecheck::Src::Inst { val, ty } = self.full.module.val(v).src {
                    args.push(ty);
                    v = val;
                }
                args.reverse();
                let mut first = true;
                for ty in args {
                    if !first {
                        write!(message, ", ").unwrap();
                    }
                    first = false;
                    write!(message, "{}", self.ty(ty)).unwrap();
                }
                write!(message, "]`").unwrap();
                emitter.diagnostic((path, range), message).finish()
            }
        }
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
