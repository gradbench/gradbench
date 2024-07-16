use std::fmt;

use crate::{
    lex::{TokenId, Tokens},
    parse::{Bind, Binop, Def, Expr, ExprId, Module, ParamId, Type, TypeId},
};

struct Printer<'a> {
    source: &'a str,
    tokens: &'a Tokens,
    module: &'a Module,
    indent: usize,
}

impl Printer<'_> {
    fn indent(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for _ in 0..self.indent {
            write!(f, "  ")?;
        }
        Ok(())
    }

    fn token(&self, f: &mut fmt::Formatter, id: TokenId) -> fmt::Result {
        write!(f, "{}", &self.source[self.tokens.get(id).byte_range()])?;
        Ok(())
    }

    fn ty(&mut self, f: &mut fmt::Formatter, id: TypeId) -> fmt::Result {
        let ty = self.module.ty(id);
        match ty {
            Type::Unit => write!(f, "()")?,
            Type::Name { name } => self.token(f, name)?,
            Type::Prod { fst, snd } => {
                self.ty(f, fst)?;
                write!(f, " * ")?;
                self.ty(f, snd)?;
            }
            Type::Sum { left, right } => {
                self.ty(f, left)?;
                write!(f, " + ")?;
                self.ty(f, right)?;
            }
            Type::Array { index, elem } => {
                write!(f, "[")?;
                self.ty(f, index)?;
                write!(f, "] ")?;
                self.ty(f, elem)?;
            }
        }
        Ok(())
    }

    fn bind(&mut self, f: &mut fmt::Formatter, bind: Bind) -> fmt::Result {
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

    fn param(&mut self, f: &mut fmt::Formatter, id: ParamId) -> fmt::Result {
        let param = self.module.param(id);
        self.bind(f, param.bind)?;
        if let Some(ty) = param.ty {
            write!(f, " : ")?;
            self.ty(f, ty)?;
        }
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
            Expr::Elem { array, index } => {
                self.expr(f, array)?;
                write!(f, "[")?;
                self.expr(f, index)?;
                write!(f, "]")?;
            }
            Expr::Apply { func, arg } => {
                write!(f, "(")?;
                self.expr(f, func)?;
                write!(f, " ")?;
                self.expr(f, arg)?;
                write!(f, ")")?;
            }
            Expr::Map { func, arg } => {
                self.expr(f, func)?;
                write!(f, ".(")?;
                self.expr(f, arg)?;
                write!(f, ")")?;
            }
            Expr::Let { param, val, body } => {
                write!(f, "let ")?;
                self.param(f, param)?;
                write!(f, " = ")?;
                if let Expr::Let { .. } = self.module.expr(val) {
                    writeln!(f, "(")?;
                    self.indent += 1;
                    self.indent(f)?;
                    self.expr(f, val)?;
                    self.indent -= 1;
                    writeln!(f)?;
                    self.indent(f)?;
                    writeln!(f, ")")?;
                } else {
                    self.expr(f, val)?;
                    writeln!(f)?;
                }
                self.indent(f)?;
                self.expr(f, body)?;
            }
            Expr::Binary { lhs, op, rhs } => {
                write!(f, "(")?;
                self.expr(f, lhs)?;
                write!(f, " ")?;
                self.binop(f, op)?;
                write!(f, " ")?;
                self.expr(f, rhs)?;
                write!(f, ")")?;
            }
            Expr::Lambda { param, ty, body } => {
                write!(f, "(")?;
                self.param(f, param)?;
                write!(f, ")")?;
                if let Some(ty) = ty {
                    write!(f, " : ")?;
                    self.ty(f, ty)?;
                }
                write!(f, " => ")?;
                self.expr(f, body)?;
            }
        }
        Ok(())
    }

    fn def(&mut self, f: &mut fmt::Formatter, def: &Def) -> fmt::Result {
        write!(f, "def ")?;
        self.token(f, def.name)?;
        for &param in &def.params {
            write!(f, " (")?;
            self.param(f, param)?;
            write!(f, ")")?;
        }
        if let Some(ty) = def.ty {
            write!(f, " : ")?;
            self.ty(f, ty)?;
        }
        writeln!(f, " =")?;
        self.indent += 1;
        self.indent(f)?;
        self.expr(f, def.body)?;
        self.indent -= 1;
        Ok(())
    }

    fn module(&mut self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;
        for def in self.module.defs() {
            if !first {
                writeln!(f)?;
            }
            first = false;
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
        indent: 0,
    }
    .module(f)
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, path::Path};

    use goldenfile::Mint;

    use crate::{lex::lex, parse::parse, util::ModuleWithSource};

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
            let module = parse(&tokens).expect(stripped);
            let pprinted = ModuleWithSource {
                source,
                tokens,
                module,
            }
            .to_string();
            let mut file = mint.new_goldenfile(stripped).expect(stripped);
            file.write_all(pprinted.as_bytes()).expect(stripped);
        }
    }
}
