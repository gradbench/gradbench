use std::{
    fmt::{self, Write},
    ops::Range,
    sync::Arc,
};

use crate::{
    graph::{Data, Graph, Uri},
    lex::{TokenId, Tokens},
    parse,
    range::{bind_range, expr_range, param_range, ty_range},
    typecheck::{self, ImportId},
    util::{Diagnostic, Emitter, Id},
};

#[derive(Clone, Debug)]
pub struct FullModule<'a> {
    pub source: &'a str,
    pub tokens: &'a Tokens,
    pub tree: &'a parse::Module,
    pub module: Arc<typecheck::Module>,
}

pub trait Importer {
    fn import(&self, id: typecheck::ImportId) -> FullModule;
}

#[derive(Clone, Debug)]
pub struct GraphImporter<'a> {
    pub graph: &'a Graph,
    pub uris: &'a [Uri],
}

impl Importer for GraphImporter<'_> {
    fn import(&self, id: ImportId) -> FullModule {
        match &self.graph.get(&self.uris[id.to_usize()]).data {
            Data::Analyzed { syn, sem, errs } => {
                assert!(errs.is_empty());
                FullModule {
                    source: &syn.src.text,
                    tokens: &syn.toks,
                    tree: &syn.tree,
                    module: Arc::clone(sem),
                }
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Printer<'a, I> {
    full: FullModule<'a>,
    import: I,
}

impl<'a, I: Clone + Importer> Printer<'a, I> {
    pub fn new(full: FullModule<'a>, import: I) -> Self {
        Self { full, import }
    }

    fn get_ty(&self, id: typecheck::TypeId) -> typecheck::Type {
        self.full.module.ty(id)
    }

    fn token_range(&self, id: TokenId) -> Range<usize> {
        self.full.tokens.get(id).byte_range()
    }

    fn ty_range(&self, id: parse::TypeId) -> Range<usize> {
        ty_range(self.full.tokens, self.full.tree, id)
    }

    fn bind_range(&self, id: parse::ParamId) -> Range<usize> {
        bind_range(self.full.tokens, self.full.tree, id)
    }

    fn param_range(&self, id: parse::ParamId) -> Range<usize> {
        param_range(self.full.tokens, self.full.tree, id)
    }

    fn expr_range(&self, id: parse::ExprId) -> Range<usize> {
        expr_range(self.full.tokens, self.full.tree, id)
    }

    fn ty(&self, id: typecheck::TypeId) -> Type<I> {
        Type {
            printer: self.clone(),
            id,
        }
    }

    fn param_ty(&self, id: parse::ParamId) -> Type<I> {
        self.ty(self.full.module.val(self.full.module.param(id)).ty)
    }

    fn expr_ty(&self, id: parse::ExprId) -> Type<I> {
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
                    Some(id) => self.import.import(id),
                    None => self.full.clone(),
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
pub struct Type<'a, I> {
    printer: Printer<'a, I>,
    id: typecheck::TypeId,
}

impl<I: Clone + Importer> fmt::Display for Type<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.printer.print_ty(f, self.id)
    }
}
