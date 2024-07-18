use serde::Serialize;
use serde_json::{json, Value};

use crate::{
    lex::{TokenId, Tokens},
    parse::{Bind, Binop, Def, Expr, ExprId, Import, Module, Param, ParamId, Type, TypeId, Unop},
};

fn array(it: impl Iterator<Item = impl Serialize>) -> Value {
    let v: Vec<_> = it.collect();
    json!(v)
}

#[derive(Debug)]
struct Json<'a> {
    source: &'a str,
    tokens: &'a Tokens,
    module: &'a Module,
}

impl<'a> Json<'a> {
    fn token(&self, id: TokenId) -> &'a str {
        &self.source[self.tokens.get(id).byte_range()]
    }

    fn ty(&self, id: TypeId) -> Value {
        match self.module.ty(id) {
            Type::Unit => json!({ "kind": "unit" }),
            Type::Name { name } => json!({ "kind": "name", "name": self.token(name) }),
            Type::Prod { fst, snd } => json!({
                "kind": "prod",
                "first": self.ty(fst),
                "second": self.ty(snd),
            }),
            Type::Sum { left, right } => json!({
                "kind": "sum",
                "left": self.ty(left),
                "right": self.ty(right),
            }),
            Type::Array { index, elem } => json!({
                "kind": "array",
                "index": index.map(|i| self.ty(i)),
                "element": self.ty(elem),
            }),
        }
    }

    fn bind(&self, bind: Bind) -> Value {
        match bind {
            Bind::Unit => json!({ "kind": "unit" }),
            Bind::Name { name } => json!({ "kind": "name", "name": self.token(name) }),
            Bind::Pair { fst, snd } => json!({
                "kind": "pair",
                "first": self.param(fst),
                "second": self.param(snd),
            }),
            Bind::Record { name, field, rest } => {
                let mut fields = vec![];
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    fields.push(json!({ "name": self.token(n), "value": self.param(v) }));
                    match self.module.param(r).bind {
                        Bind::Record { name, field, rest } => {
                            (n, v, r) = (name, field, rest);
                        }
                        Bind::End => break,
                        _ => panic!("invalid record"),
                    }
                }
                json!({ "kind": "record", "fields": fields })
            }
            Bind::End => json!({ "kind": "record", "fields": [] }),
        }
    }

    fn param(&self, id: ParamId) -> Value {
        let Param { bind, ty } = self.module.param(id);
        let bind = self.bind(bind);
        json!({ "bind": bind, "type": ty.map(|t| self.ty(t)) })
    }

    fn unop(&self, op: Unop) -> Value {
        match op {
            Unop::Neg => json!("neg"),
        }
    }

    fn binop(&self, op: Binop) -> Value {
        match op {
            Binop::Add => json!("add"),
            Binop::Sub => json!("sub"),
            Binop::Mul => json!("mul"),
            Binop::Div => json!("div"),
        }
    }

    fn expr(&self, id: ExprId) -> Value {
        match self.module.expr(id) {
            Expr::Name { name } => json!({ "kind": "name", "name": self.token(name) }),
            Expr::Unit => json!({ "kind": "unit" }),
            Expr::Number { val } => {
                let x = self.tokens.get(val).number(self.source);
                json!({ "kind": "number", "value": x })
            }
            Expr::Pair { fst, snd } => json!({
                "kind": "pair",
                "first": self.expr(fst),
                "second": self.expr(snd),
            }),
            Expr::Record { name, field, rest } => {
                let mut fields = vec![];
                let (mut n, mut f, mut r) = (name, field, rest);
                loop {
                    fields.push(json!({ "name": self.token(n), "value": self.expr(f) }));
                    match self.module.expr(r) {
                        Expr::Record { name, field, rest } => {
                            (n, f, r) = (name, field, rest);
                        }
                        Expr::End => break,
                        _ => panic!("invalid record"),
                    }
                }
                json!({ "kind": "record", "fields": fields })
            }
            Expr::End => json!({ "kind": "record", "fields": [] }),
            Expr::Elem { array, index } => json!({
                "kind": "element",
                "array": self.expr(array),
                "index": self.expr(index),
            }),
            Expr::Apply { func, arg } => json!({
                "kind": "apply",
                "function": self.expr(func),
                "argument": self.expr(arg),
            }),
            Expr::Map { func, arg } => json!({
                "kind": "map",
                "function": self.expr(func),
                "argument": self.expr(arg),
            }),
            Expr::Let { param, val, body } => json!({
                "kind": "let",
                "parameter": self.param(param),
                "value": self.expr(val),
                "body": self.expr(body),
            }),
            Expr::Index { name, val, body } => json!({
                "kind": "index",
                "name": self.token(name),
                "value": self.expr(val),
                "body": self.expr(body),
            }),
            Expr::Unary { op, arg } => json!({
                "kind": "unary",
                "operator": self.unop(op),
                "argument": self.expr(arg),
            }),
            Expr::Binary { lhs, map, op, rhs } => json!({
                "kind": "binary",
                "left": self.expr(lhs),
                "map": map,
                "operator": self.binop(op),
                "right": self.expr(rhs),
            }),
            Expr::Lambda { param, ty, body } => json!({
                "kind": "lambda",
                "parameter": self.param(param),
                "return": ty.map(|t| self.ty(t)),
                "body": self.expr(body),
            }),
        }
    }

    fn import(&self, Import { module, names }: &Import) -> Value {
        let name = self.tokens.get(*module).string(self.source);
        let uses = array(names.iter().map(|name| self.token(*name)));
        json!({ "name": name, "use": uses })
    }

    fn def(
        &self,
        Def {
            name,
            types,
            params,
            ty,
            body,
        }: &Def,
    ) -> Value {
        let types = array(types.iter().map(|ty| self.token(*ty)));
        let params = array(params.iter().map(|param| self.param(*param)));
        json!({
            "name": self.token(*name),
            "types": types,
            "parameters": params,
            "return": ty.map(|t| self.ty(t)),
            "body": self.expr(*body),
        })
    }

    fn module(&self) -> Value {
        let imports = array(
            self.module
                .imports()
                .iter()
                .map(|import| self.import(import)),
        );
        let defs = array(self.module.defs().iter().map(|def| self.def(def)));
        json!({ "imports": imports, "definitions": defs })
    }
}

pub fn json(source: &str, tokens: &Tokens, module: &Module) -> Value {
    Json {
        source,
        tokens,
        module,
    }
    .module()
}
