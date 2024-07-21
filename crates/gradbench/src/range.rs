use std::ops::Range;

use crate::{
    lex::{TokenId, TokenKind, Tokens},
    parse::{Bind, Expr, ExprId, Module, Param, ParamId, Type, TypeId},
};

#[derive(Debug)]
struct Ranger<'a> {
    tokens: &'a Tokens,
    tree: &'a Module,
}

impl Ranger<'_> {
    fn before(&self, mut token: TokenId) -> TokenId {
        loop {
            assert!(token.index > 0);
            token.index -= 1;
            if !self.tokens.get(token).kind.ignore() {
                return token;
            }
        }
    }

    fn after(&self, mut token: TokenId) -> TokenId {
        loop {
            assert_ne!(self.tokens.get(token).kind, TokenKind::Eof);
            token.index += 1;
            if !self.tokens.get(token).kind.ignore() {
                return token;
            }
        }
    }

    fn ty_start(&self, ty: TypeId) -> TokenId {
        match self.tree.ty(ty) {
            Type::Paren { inner } => self.before(self.ty_start(inner)),
            Type::Unit { open, close: _ } => open,
            Type::Name { name } => name,
            Type::Prod { fst, snd: _ } => self.ty_start(fst),
            Type::Sum { left, right: _ } => self.ty_start(left),
            Type::Array { index, elem } => match index {
                Some(i) => self.before(self.ty_start(i)),
                None => self.before(self.before(self.ty_start(elem))),
            },
            Type::Func { dom, cod: _ } => self.ty_start(dom),
        }
    }

    fn ty_end(&self, ty: TypeId) -> TokenId {
        match self.tree.ty(ty) {
            Type::Paren { inner } => self.after(self.ty_end(inner)),
            Type::Unit { open: _, close } => close,
            Type::Name { name } => name,
            Type::Prod { fst: _, snd } => self.ty_end(snd),
            Type::Sum { left: _, right } => self.ty_end(right),
            Type::Array { index: _, elem } => self.ty_end(elem),
            Type::Func { dom: _, cod } => self.ty_end(cod),
        }
    }

    fn bind_start(&self, bind: Bind) -> TokenId {
        match bind {
            Bind::Paren { inner } => self.before(self.param_start(inner)),
            Bind::Unit { open, close: _ } => open,
            Bind::Name { name } => name,
            Bind::Pair { fst, snd: _ } => self.param_start(fst),
            Bind::Record {
                name: _,
                field: _,
                rest,
            } => self.param_start(rest),
            Bind::End { open, close: _ } => open,
        }
    }

    fn bind_end(&self, bind: Bind) -> TokenId {
        match bind {
            Bind::Paren { inner } => self.after(self.param_end(inner)),
            Bind::Unit { open: _, close } => close,
            Bind::Name { name } => name,
            Bind::Pair { fst: _, snd } => self.param_end(snd),
            Bind::Record {
                name: _,
                field: _,
                rest,
            } => self.param_end(rest),
            Bind::End { open: _, close } => close,
        }
    }

    fn param_start(&self, param: ParamId) -> TokenId {
        let Param { bind, ty: _ } = self.tree.param(param);
        self.bind_start(bind)
    }

    fn param_end(&self, param: ParamId) -> TokenId {
        let Param { bind, ty } = self.tree.param(param);
        match ty {
            Some(t) => self.ty_end(t),
            None => self.bind_end(bind),
        }
    }

    fn expr_start(&self, expr: ExprId) -> TokenId {
        match self.tree.expr(expr) {
            Expr::Paren { inner } => self.before(self.expr_start(inner)),
            Expr::Name { name } => name,
            Expr::Unit { open, close: _ } => open,
            Expr::Number { val } => val,
            Expr::Pair { fst, snd: _ } => self.expr_start(fst),
            Expr::Record {
                name: _,
                field: _,
                rest,
            } => self.expr_start(rest),
            Expr::End { open, close: _ } => open,
            Expr::Elem { array, index: _ } => self.expr_start(array),
            Expr::Apply { func, arg: _ } => self.expr_start(func),
            Expr::Map { func, arg: _ } => self.expr_start(func),
            Expr::Let {
                param,
                val: _,
                body: _,
            } => self.before(self.param_start(param)),
            Expr::Index {
                name,
                val: _,
                body: _,
            } => self.before(name),
            Expr::Unary { op: _, arg } => self.before(self.expr_start(arg)),
            Expr::Binary {
                lhs,
                map: _,
                op: _,
                rhs: _,
            } => self.expr_start(lhs),
            Expr::Lambda {
                param,
                ty: _,
                body: _,
            } => self.param_start(param),
        }
    }

    fn expr_end(&self, expr: ExprId) -> TokenId {
        match self.tree.expr(expr) {
            Expr::Paren { inner } => self.after(self.expr_end(inner)),
            Expr::Name { name } => name,
            Expr::Unit { open: _, close } => close,
            Expr::Number { val } => val,
            Expr::Pair { fst: _, snd } => self.expr_end(snd),
            Expr::Record {
                name: _,
                field: _,
                rest,
            } => self.expr_end(rest),
            Expr::End { open: _, close } => close,
            Expr::Elem { array: _, index } => self.after(self.expr_end(index)),
            Expr::Apply { func: _, arg } => self.expr_end(arg),
            Expr::Map { func: _, arg } => self.after(self.expr_end(arg)),
            Expr::Let {
                param: _,
                val: _,
                body,
            } => self.expr_end(body),
            Expr::Index {
                name: _,
                val: _,
                body,
            } => self.expr_end(body),
            Expr::Unary { op: _, arg } => self.expr_end(arg),
            Expr::Binary {
                lhs: _,
                map: _,
                op: _,
                rhs,
            } => self.expr_end(rhs),
            Expr::Lambda {
                param: _,
                ty: _,
                body,
            } => self.expr_end(body),
        }
    }
}

pub fn ty_range(tokens: &Tokens, tree: &Module, ty: TypeId) -> Range<usize> {
    let tree_with_tokens = Ranger { tokens, tree };
    let a = tokens.get(tree_with_tokens.ty_start(ty)).byte_range();
    let b = tokens.get(tree_with_tokens.ty_end(ty)).byte_range();
    a.start..b.end
}

pub fn param_range(tokens: &Tokens, tree: &Module, param: ParamId) -> Range<usize> {
    let tree_with_tokens = Ranger { tokens, tree };
    let a = tokens.get(tree_with_tokens.param_start(param)).byte_range();
    let b = tokens.get(tree_with_tokens.param_end(param)).byte_range();
    a.start..b.end
}

pub fn expr_range(tokens: &Tokens, tree: &Module, id: ExprId) -> Range<usize> {
    let tree_with_tokens = Ranger { tokens, tree };
    let a = tokens.get(tree_with_tokens.expr_start(id)).byte_range();
    let b = tokens.get(tree_with_tokens.expr_end(id)).byte_range();
    a.start..b.end
}
