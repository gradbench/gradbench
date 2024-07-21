use std::fmt;

use crate::{lex, typecheck};

pub fn u32_to_usize(n: u32) -> usize {
    n.try_into()
        .expect("pointer size is assumed to be at least 32 bits")
}

#[derive(Clone, Copy, Debug)]
pub struct Printer<'a> {
    pub source: &'a str,
    pub tokens: &'a lex::Tokens,
    pub module: &'a typecheck::Module,
}

impl Printer<'_> {
    pub fn ty(&self, id: typecheck::TypeId) -> Type {
        Type { printer: *self, id }
    }

    pub fn print_ty(&self, w: &mut impl fmt::Write, id: typecheck::TypeId) -> fmt::Result {
        use typecheck::Type::*;
        match self.module.ty(id) {
            Untyped => write!(w, "?")?,
            Var { src, def } => {
                assert!(src.is_none(), "printed type var source should be local");
                write!(w, "{}", &self.source[self.tokens.get(def).byte_range()])?;
            }
            Unit => write!(w, "()")?,
            Int => write!(w, "Int")?,
            Float => write!(w, "Float")?,
            Prod { fst, snd } => {
                if let Prod { .. } | Sum { .. } | Func { .. } = self.module.ty(fst) {
                    write!(w, "({})", self.ty(fst))
                } else {
                    write!(w, "{}", self.ty(fst))
                }?;
                write!(w, " * ")?;
                if let Sum { .. } | Func { .. } = self.module.ty(snd) {
                    write!(w, "({})", self.ty(snd))
                } else {
                    write!(w, "{}", self.ty(snd))
                }?;
            }
            Sum { left, right } => {
                if let Prod { .. } | Sum { .. } | Func { .. } = self.module.ty(left) {
                    write!(w, "({})", self.ty(left))
                } else {
                    write!(w, "{}", self.ty(left))
                }?;
                write!(w, " + ")?;
                if let Func { .. } = self.module.ty(right) {
                    write!(w, "({})", self.ty(right))
                } else {
                    write!(w, "{}", self.ty(right))
                }?;
            }
            Array { index, elem } => {
                write!(w, "[{}]", self.ty(index))?;
                if let Prod { .. } | Sum { .. } | Func { .. } = self.module.ty(elem) {
                    write!(w, "({})", self.ty(elem))
                } else {
                    write!(w, "{}", self.ty(elem))
                }?;
            }
            Record { name, field, rest } => {
                write!(w, "{{")?;
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    write!(w, "{}: {}", self.module.field(n), self.ty(v))?;
                    match self.module.ty(r) {
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
                if let Func { .. } = self.module.ty(dom) {
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
