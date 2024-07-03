use std::fmt;

use crate::{
    lex::{TokenId, Tokens},
    parse::{Bind, BindId, Binop, Def, Expr, ExprId, Module, Param, Type, TypeId},
};

struct Printer<'a> {
    source: &'a str,
    tokens: &'a Tokens,
    module: &'a Module,
}

impl Printer<'_> {
    fn token(&self, f: &mut fmt::Formatter, id: TokenId) -> fmt::Result {
        write!(f, "{}", &self.source[self.tokens.get(id).byte_range()])?;
        Ok(())
    }

    fn ty(&mut self, f: &mut fmt::Formatter, id: TypeId) -> fmt::Result {
        let ty = self.module.ty(id);
        match ty {
            Type::Unit => write!(f, "()")?,
            Type::Name { name } => self.token(f, name)?,
            Type::Pair { fst, snd } => {
                write!(f, "(")?;
                self.ty(f, fst)?;
                write!(f, ", ")?;
                self.ty(f, snd)?;
                write!(f, ")")?;
            }
        }
        Ok(())
    }

    fn bind(&mut self, f: &mut fmt::Formatter, id: BindId) -> fmt::Result {
        let bind = self.module.bind(id);
        match bind {
            Bind::Unit => write!(f, "()")?,
            Bind::Name { name } => self.token(f, name)?,
            Bind::Pair { fst, snd } => {
                write!(f, "(")?;
                self.param(f, fst)?;
                write!(f, ", ")?;
                self.param(f, snd)?;
                write!(f, ")")?;
            }
        }
        Ok(())
    }

    fn param(&mut self, f: &mut fmt::Formatter, param: Param) -> fmt::Result {
        write!(f, "(")?;
        self.bind(f, param.bind)?;
        if let Some(ty) = param.ty {
            write!(f, " : ")?;
            self.ty(f, ty)?;
        }
        write!(f, ")")?;
        Ok(())
    }

    fn binop(&mut self, f: &mut fmt::Formatter, op: Binop) -> fmt::Result {
        let s = match op {
            Binop::Add => "+",
            Binop::Sub => "-",
            Binop::Mul => "*",
            Binop::Div => "/",
        };
        write!(f, "{}", s)?;
        Ok(())
    }

    fn expr(&mut self, f: &mut fmt::Formatter, id: ExprId) -> fmt::Result {
        let expr = self.module.expr(id);
        match expr {
            Expr::Name { name } => self.token(f, name)?,
            Expr::Unit => write!(f, "()")?,
            Expr::Number { val } => self.token(f, val)?,
            Expr::Pair { fst, snd } => {
                write!(f, "(")?;
                self.expr(f, fst)?;
                write!(f, ", ")?;
                self.expr(f, snd)?;
                write!(f, ")")?;
            }
            Expr::Apply { func, arg } => {
                self.expr(f, func)?;
                write!(f, " ")?;
                self.expr(f, arg)?;
            }
            Expr::Let { param, val, body } => {
                write!(f, "let ")?;
                self.param(f, param)?;
                write!(f, " = ")?;
                self.expr(f, val)?;
                write!(f, "; ")?;
                self.expr(f, body)?;
            }
            Expr::Binary { lhs, op, rhs } => {
                self.expr(f, lhs)?;
                write!(f, " ")?;
                self.binop(f, op)?;
                write!(f, " ")?;
                self.expr(f, rhs)?;
            }
        }
        Ok(())
    }

    fn def(&mut self, f: &mut fmt::Formatter, def: &Def) -> fmt::Result {
        write!(f, "def ")?;
        self.token(f, def.name)?;
        for &param in &def.params {
            write!(f, " ")?;
            self.param(f, param)?;
        }
        if let Some(ty) = def.ty {
            write!(f, " : ")?;
            self.ty(f, ty)?;
        }
        write!(f, " = ")?;
        self.expr(f, def.body)?;
        Ok(())
    }

    fn module(&mut self, f: &mut fmt::Formatter) -> fmt::Result {
        self.def(f, &self.module.defs()[0])?;
        writeln!(f)?;
        for def in self.module.defs().iter().skip(1) {
            writeln!(f)?;
            self.def(f, def)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn pprint(
    f: &mut fmt::Formatter,
    source: &str,
    tokens: &Tokens,
    module: &Module,
) -> fmt::Result {
    Printer {
        source,
        tokens,
        module,
    }
    .module(f)
}
