use std::{cell::Cell, cmp::Ordering, ops::Range};

use crate::{
    lex::{TokenId, TokenKind, Tokens},
    parse::{Bind, Expr, ExprId, Module, Param, ParamId, Type, TypeId},
    util::Id,
};

#[derive(Debug)]
struct Cache {
    eof: TokenId,
    ranges: Box<[(Cell<TokenId>, Cell<TokenId>)]>,
}

impl Cache {
    fn new<T>(eof: TokenId, nodes: &[T]) -> Self {
        Self {
            eof,
            ranges: vec![(Cell::new(eof), Cell::new(eof)); nodes.len()].into_boxed_slice(),
        }
    }

    fn check(&self, id: TokenId) -> Option<TokenId> {
        if id == self.eof {
            None
        } else {
            Some(id)
        }
    }
}

fn get_start(cache: Option<&Cache>, id: impl Id) -> Option<TokenId> {
    cache.as_ref().and_then(|c| {
        let (start, _) = &c.ranges[id.to_usize()];
        c.check(start.get())
    })
}

fn get_end(cache: Option<&Cache>, id: impl Id) -> Option<TokenId> {
    cache.as_ref().and_then(|c| {
        let (_, end) = &c.ranges[id.to_usize()];
        c.check(end.get())
    })
}

fn put_start(cache: Option<&Cache>, id: impl Id, tok: TokenId) -> TokenId {
    if let Some(c) = cache {
        let (start, _) = &c.ranges[id.to_usize()];
        start.set(tok);
    }
    tok
}

fn put_end(cache: Option<&Cache>, id: impl Id, tok: TokenId) -> TokenId {
    if let Some(c) = cache {
        let (_, end) = &c.ranges[id.to_usize()];
        end.set(tok);
    }
    tok
}

#[derive(Debug)]
struct Caches {
    ty: Cache,
    param: Cache,
    expr: Cache,
}

#[derive(Debug)]
struct Ranger<'a> {
    tokens: &'a Tokens,
    tree: &'a Module,
    cache: Option<Caches>,
}

impl<'a> Ranger<'a> {
    fn new(tokens: &'a Tokens, tree: &'a Module) -> Self {
        Self {
            tokens,
            tree,
            cache: None,
        }
    }

    fn with_cache(tokens: &'a Tokens, tree: &'a Module) -> Self {
        let mut eof = TokenId { index: 0 };
        while tokens.get(eof).kind != TokenKind::Eof {
            eof.index += 1;
        }
        Self {
            tokens,
            tree,
            cache: Some(Caches {
                ty: Cache::new(eof, tree.types()),
                param: Cache::new(eof, tree.params()),
                expr: Cache::new(eof, tree.exprs()),
            }),
        }
    }

    fn range(&self, start: TokenId, end: TokenId) -> Range<usize> {
        let a = self.tokens.get(start).byte_range();
        let b = self.tokens.get(end).byte_range();
        a.start..b.end
    }

    fn types(&self) -> Option<&Cache> {
        self.cache.as_ref().map(|c| &c.ty)
    }

    fn params(&self) -> Option<&Cache> {
        self.cache.as_ref().map(|c| &c.param)
    }

    fn exprs(&self) -> Option<&Cache> {
        self.cache.as_ref().map(|c| &c.expr)
    }

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
        let types = self.types();
        if let Some(tok) = get_start(types, ty) {
            return tok;
        }
        let tok = match self.tree.ty(ty) {
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
        };
        put_start(types, ty, tok)
    }

    fn ty_end(&self, ty: TypeId) -> TokenId {
        let types = self.types();
        if let Some(tok) = get_end(types, ty) {
            return tok;
        }
        let tok = match self.tree.ty(ty) {
            Type::Paren { inner } => self.after(self.ty_end(inner)),
            Type::Unit { open: _, close } => close,
            Type::Name { name } => name,
            Type::Prod { fst: _, snd } => self.ty_end(snd),
            Type::Sum { left: _, right } => self.ty_end(right),
            Type::Array { index: _, elem } => self.ty_end(elem),
            Type::Func { dom: _, cod } => self.ty_end(cod),
        };
        put_end(types, ty, tok)
    }

    fn ty_range(&self, ty: TypeId) -> Range<usize> {
        self.range(self.ty_start(ty), self.ty_end(ty))
    }

    fn bind_start(&self, bind: Bind) -> TokenId {
        match bind {
            Bind::Paren { inner } => self.before(self.param_start(inner)),
            Bind::Unit { open, close: _ } => open,
            Bind::Name { name } => name,
            Bind::Pair { fst, snd: _ } => self.param_start(fst),
            Bind::Record {
                name,
                field: _,
                rest: _,
            } => name,
            Bind::End { open, close } => match self.tokens.get(self.before(close)).kind {
                TokenKind::LBrace => open,
                _ => close,
            },
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

    fn bind_range(&self, bind: Bind) -> Range<usize> {
        self.range(self.bind_start(bind), self.bind_end(bind))
    }

    fn param_start(&self, param: ParamId) -> TokenId {
        let params = self.params();
        if let Some(tok) = get_start(params, param) {
            return tok;
        }
        let Param { bind, ty: _ } = self.tree.param(param);
        let tok = self.bind_start(bind);
        put_start(params, param, tok)
    }

    fn param_end(&self, param: ParamId) -> TokenId {
        let params = self.params();
        if let Some(tok) = get_end(params, param) {
            return tok;
        }
        let Param { bind, ty } = self.tree.param(param);
        let tok = match ty {
            Some(t) => self.ty_end(t),
            None => self.bind_end(bind),
        };
        put_end(params, param, tok)
    }

    fn param_range(&self, param: ParamId) -> Range<usize> {
        self.range(self.param_start(param), self.param_end(param))
    }

    fn expr_start(&self, expr: ExprId) -> TokenId {
        let exprs = self.exprs();
        if let Some(tok) = get_start(exprs, expr) {
            return tok;
        }
        let tok = match self.tree.expr(expr) {
            Expr::Paren { inner } => self.before(self.expr_start(inner)),
            Expr::Name { name } => name,
            Expr::Undefined { token } => token,
            Expr::Unit { open, close: _ } => open,
            Expr::Number { val } => val,
            Expr::Pair { fst, snd: _ } => self.expr_start(fst),
            Expr::Record {
                name,
                field: _,
                rest: _,
            } => name,
            Expr::End { open, close } => match self.tokens.get(self.before(close)).kind {
                TokenKind::LBrace => open,
                _ => close,
            },
            Expr::Elem { array, index: _ } => self.expr_start(array),
            Expr::Inst { val, ty: _ } => self.expr_start(val),
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
            Expr::Binary { lhs, op: _, rhs: _ } => self.expr_start(lhs),
            Expr::Lambda {
                param,
                ty: _,
                body: _,
            } => self.param_start(param),
        };
        put_start(exprs, expr, tok)
    }

    fn expr_end(&self, expr: ExprId) -> TokenId {
        let exprs = self.exprs();
        let tok = match self.tree.expr(expr) {
            Expr::Paren { inner } => self.after(self.expr_end(inner)),
            Expr::Name { name } => name,
            Expr::Undefined { token } => token,
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
            Expr::Inst { val: _, ty } => self.after(self.ty_end(ty)),
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
            Expr::Binary { lhs: _, op: _, rhs } => self.expr_end(rhs),
            Expr::Lambda {
                param: _,
                ty: _,
                body,
            } => self.expr_end(body),
        };
        put_end(exprs, expr, tok)
    }

    fn expr_range(&self, expr: ExprId) -> Range<usize> {
        self.range(self.expr_start(expr), self.expr_end(expr))
    }
}

pub fn ty_range(tokens: &Tokens, tree: &Module, id: TypeId) -> Range<usize> {
    Ranger::new(tokens, tree).ty_range(id)
}

pub fn bind_range(tokens: &Tokens, tree: &Module, id: ParamId) -> Range<usize> {
    let Param { bind, ty: _ } = tree.param(id);
    Ranger::new(tokens, tree).bind_range(bind)
}

pub fn param_range(tokens: &Tokens, tree: &Module, id: ParamId) -> Range<usize> {
    Ranger::new(tokens, tree).param_range(id)
}

pub fn expr_range(tokens: &Tokens, tree: &Module, id: ExprId) -> Range<usize> {
    Ranger::new(tokens, tree).expr_range(id)
}

#[derive(Debug)]
pub enum Node {
    Type(TypeId),
    Param(ParamId),
    Expr(ExprId),
}

pub fn find(tokens: &Tokens, tree: &Module, offset: usize) -> Option<(Node, Range<usize>)> {
    let ranger = Ranger::with_cache(tokens, tree);
    (tree.types().iter().enumerate().map(|(i, _)| {
        let id = TypeId::from_usize(i).unwrap();
        (Node::Type(id), ranger.ty_range(id))
    }))
    .chain(tree.params().iter().enumerate().map(|(i, _)| {
        let id = ParamId::from_usize(i).unwrap();
        (Node::Param(id), ranger.param_range(id))
    }))
    .chain(tree.exprs().iter().enumerate().map(|(i, _)| {
        let id = ExprId::from_usize(i).unwrap();
        (Node::Expr(id), ranger.expr_range(id))
    }))
    .filter(|(_, range)| range.contains(&offset))
    .min_by(|(_, r1), (_, r2)| {
        if r1 == r2 {
            Ordering::Equal
        } else if r2.start <= r1.start && r1.end <= r2.end {
            Ordering::Less
        } else if r1.start <= r2.start && r2.end <= r1.end {
            Ordering::Greater
        } else {
            panic!("incomparable node ranges")
        }
    })
}
