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
        match self.module.ty(id) {
            typecheck::Type::Var { src, def } => todo!(),
            typecheck::Type::Unit => todo!(),
            typecheck::Type::Int => todo!(),
            typecheck::Type::Float => write!(w, "Float"),
            typecheck::Type::Prod { fst, snd } => todo!(),
            typecheck::Type::Sum { left, right } => todo!(),
            typecheck::Type::Array { index, elem } => todo!(),
            typecheck::Type::Record { name, field, rest } => todo!(),
            typecheck::Type::End => todo!(),
            typecheck::Type::Func { dom, cod } => write!(w, "{} -> {}", self.ty(dom), self.ty(cod)),
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
