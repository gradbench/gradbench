use std::fmt;

use crate::{
    lex::{TokenId, Tokens},
    parse::{Bind, Binop, Def, Expr, ExprId, Import, Module, Param, ParamId, Type, TypeId, Unop},
};

#[derive(Debug)]
struct Printer<'a> {
    source: &'a str,
    tokens: &'a Tokens,
    tree: &'a Module,
    indent: usize,
}

impl Printer<'_> {
    fn indent(&self, w: &mut impl fmt::Write) -> fmt::Result {
        for _ in 0..self.indent {
            write!(w, "  ")?;
        }
        Ok(())
    }

    fn token(&self, w: &mut impl fmt::Write, id: TokenId) -> fmt::Result {
        write!(w, "{}", &self.source[self.tokens.get(id).byte_range()])?;
        Ok(())
    }

    fn ty(&mut self, w: &mut impl fmt::Write, id: TypeId) -> fmt::Result {
        let ty = self.tree.ty(id);
        match ty {
            Type::Paren { inner } => {
                write!(w, "(")?;
                self.ty(w, inner)?;
                write!(w, ")")?;
            }
            Type::Unit { open, close } => {
                self.token(w, open)?;
                self.token(w, close)?;
            }
            Type::Name { name } => self.token(w, name)?,
            Type::Prod { fst, snd } => {
                self.ty(w, fst)?;
                write!(w, " * ")?;
                self.ty(w, snd)?;
            }
            Type::Sum { left, right } => {
                self.ty(w, left)?;
                write!(w, " + ")?;
                self.ty(w, right)?;
            }
            Type::Array { index, elem } => {
                write!(w, "[")?;
                if let Some(i) = index {
                    self.ty(w, i)?;
                }
                write!(w, "]")?;
                self.ty(w, elem)?;
            }
            Type::Func { dom, cod } => {
                self.ty(w, dom)?;
                write!(w, " -> ")?;
                self.ty(w, cod)?;
            }
        }
        Ok(())
    }

    fn bind(&mut self, w: &mut impl fmt::Write, bind: Bind) -> fmt::Result {
        match bind {
            Bind::Paren { inner } => {
                write!(w, "(")?;
                self.param(w, inner)?;
                write!(w, ")")?;
            }
            Bind::Unit { open, close } => {
                self.token(w, open)?;
                self.token(w, close)?;
            }
            Bind::Name { name } => self.token(w, name)?,
            Bind::Pair { fst, snd } => {
                self.param(w, fst)?;
                write!(w, ", ")?;
                self.param(w, snd)?;
            }
            Bind::Record { name, field, rest } => {
                write!(w, "{{")?;
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    self.token(w, n)?;
                    write!(w, " = ")?;
                    self.param(w, v)?;
                    let Param { bind, ty } = self.tree.param(r);
                    assert_eq!(ty, None);
                    match bind {
                        Bind::Record { name, field, rest } => {
                            write!(w, ", ")?;
                            (n, v, r) = (name, field, rest);
                        }
                        Bind::End { open: _, close: _ } => break,
                        _ => panic!("invalid record"),
                    }
                }
                write!(w, "}}")?;
            }
            Bind::End { open, close } => {
                self.token(w, open)?;
                self.token(w, close)?;
            }
        }
        Ok(())
    }

    fn param(&mut self, w: &mut impl fmt::Write, id: ParamId) -> fmt::Result {
        let param = self.tree.param(id);
        self.bind(w, param.bind)?;
        if let Some(ty) = param.ty {
            write!(w, " : ")?;
            self.ty(w, ty)?;
        }
        Ok(())
    }

    fn unop(&mut self, w: &mut impl fmt::Write, op: Unop) -> fmt::Result {
        let s = match op {
            Unop::Neg => "-",
        };
        write!(w, "{}", s)?;
        Ok(())
    }

    fn binop(&mut self, w: &mut impl fmt::Write, op: Binop) -> fmt::Result {
        let s = match op {
            Binop::Add => "+",
            Binop::Sub => "-",
            Binop::Mul => "*",
            Binop::Div => "/",
        };
        write!(w, "{}", s)?;
        Ok(())
    }

    fn expr(&mut self, w: &mut impl fmt::Write, id: ExprId) -> fmt::Result {
        match self.tree.expr(id) {
            Expr::Paren { inner } => {
                write!(w, "(")?;
                if let Expr::Let { .. } | Expr::Index { .. } = self.tree.expr(inner) {
                    writeln!(w)?;
                    self.indent += 1;
                    self.indent(w)?;
                    self.expr(w, inner)?;
                    self.indent -= 1;
                    writeln!(w)?;
                    self.indent(w)?;
                } else {
                    self.expr(w, inner)?;
                }
                write!(w, ")")?;
            }
            Expr::Name { name } => self.token(w, name)?,
            Expr::Undefined { token } => self.token(w, token)?,
            Expr::Unit { open, close } => {
                self.token(w, open)?;
                self.token(w, close)?;
            }
            Expr::Number { val } => self.token(w, val)?,
            Expr::Pair { fst, snd } => {
                self.expr(w, fst)?;
                write!(w, ", ")?;
                self.expr(w, snd)?;
            }
            Expr::Record { name, field, rest } => {
                write!(w, "{{")?;
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    self.token(w, n)?;
                    write!(w, " = ")?;
                    self.expr(w, v)?;
                    match self.tree.expr(r) {
                        Expr::Record { name, field, rest } => {
                            write!(w, ", ")?;
                            (n, v, r) = (name, field, rest);
                        }
                        Expr::End { open: _, close: _ } => break,
                        _ => panic!("invalid record"),
                    }
                }
                write!(w, "}}")?;
            }
            Expr::End { open, close } => {
                self.token(w, open)?;
                self.token(w, close)?;
            }
            Expr::Elem { array, index } => {
                self.expr(w, array)?;
                write!(w, "[")?;
                self.expr(w, index)?;
                write!(w, "]")?;
            }
            Expr::Apply { func, arg } => {
                self.expr(w, func)?;
                write!(w, " ")?;
                self.expr(w, arg)?;
            }
            Expr::Map { func, arg } => {
                self.expr(w, func)?;
                write!(w, ".(")?;
                self.expr(w, arg)?;
                write!(w, ")")?;
            }
            Expr::Let { param, val, body } => {
                write!(w, "let ")?;
                self.param(w, param)?;
                write!(w, " = ")?;
                self.expr(w, val)?;
                writeln!(w)?;
                self.indent(w)?;
                self.expr(w, body)?;
            }
            Expr::Index { name, val, body } => {
                write!(w, "index ")?;
                self.token(w, name)?;
                write!(w, " <- ")?;
                self.expr(w, val)?;
                writeln!(w)?;
                self.indent(w)?;
                self.expr(w, body)?;
            }
            Expr::Unary { op, arg } => {
                self.unop(w, op)?;
                self.expr(w, arg)?;
            }
            Expr::Binary { lhs, map, op, rhs } => {
                self.expr(w, lhs)?;
                write!(w, " ")?;
                if map {
                    write!(w, ".")?;
                }
                self.binop(w, op)?;
                write!(w, " ")?;
                self.expr(w, rhs)?;
            }
            Expr::Lambda { param, ty, body } => {
                self.param(w, param)?;
                if let Some(ty) = ty {
                    write!(w, " : ")?;
                    self.ty(w, ty)?;
                }
                write!(w, " => ")?;
                self.expr(w, body)?;
            }
        }
        Ok(())
    }

    fn import(&mut self, w: &mut impl fmt::Write, import: &Import) -> fmt::Result {
        let Import { module, names } = import;
        write!(w, "import ")?;
        self.token(w, *module)?;
        write!(w, " use ")?;
        let mut first = true;
        for &name in names {
            if !first {
                write!(w, ", ")?;
            }
            first = false;
            self.token(w, name)?;
        }
        writeln!(w)?;
        Ok(())
    }

    fn def(&mut self, w: &mut impl fmt::Write, def: &Def) -> fmt::Result {
        let Def {
            name,
            types,
            params,
            ty,
            body,
        } = def;
        write!(w, "def ")?;
        self.token(w, *name)?;
        if !types.is_empty() {
            let mut first = true;
            write!(w, " {{")?;
            for &t in types {
                if !first {
                    write!(w, ", ")?;
                }
                first = false;
                self.token(w, t)?;
            }
            write!(w, "}}")?;
        }
        for &param in params {
            write!(w, " (")?;
            self.param(w, param)?;
            write!(w, ")")?;
        }
        if let Some(ty) = ty {
            write!(w, " : ")?;
            self.ty(w, *ty)?;
        }
        writeln!(w, " =")?;
        self.indent += 1;
        self.indent(w)?;
        self.expr(w, *body)?;
        self.indent -= 1;
        Ok(())
    }

    fn module(&mut self, w: &mut impl fmt::Write) -> fmt::Result {
        let mut first = true;
        for import in self.tree.imports() {
            first = false;
            self.import(w, import)?;
        }
        for def in self.tree.defs() {
            if !first {
                writeln!(w)?;
            }
            first = false;
            self.def(w, def)?;
            writeln!(w)?;
        }
        Ok(())
    }
}

pub fn pprint(
    w: &mut impl fmt::Write,
    source: &str,
    tokens: &Tokens,
    tree: &Module,
) -> fmt::Result {
    Printer {
        source,
        tokens,
        tree,
        indent: 0,
    }
    .module(w)
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, path::Path};

    use goldenfile::Mint;

    use crate::{lex::lex, parse::parse};

    use super::*;

    #[test]
    fn test_examples() {
        let prefix = Path::new("src/pprint");
        let input = prefix.join("input");
        let mut mint = Mint::new(prefix.join("output"));
        for entry in fs::read_dir(&input).unwrap() {
            let path = entry.unwrap().path();
            let stripped = path.strip_prefix(&input).unwrap().to_str().unwrap();
            let source = fs::read_to_string(&path).expect(stripped);
            let tokens = lex(&source).expect(stripped);
            let tree = parse(&tokens).expect(stripped);
            let mut pprinted = String::new();
            pprint(&mut pprinted, &source, &tokens, &tree).unwrap();
            let mut file = mint.new_goldenfile(stripped).expect(stripped);
            file.write_all(pprinted.as_bytes()).expect(stripped);
        }
    }
}
