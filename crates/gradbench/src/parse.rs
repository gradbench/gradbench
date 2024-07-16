use enumset::EnumSet;

use crate::{
    lex::{
        TokenId,
        TokenKind::{self, *},
        Tokens,
    },
    util::u32_to_usize,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TypeId {
    pub index: u32,
}

impl From<TypeId> for usize {
    fn from(id: TypeId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParamId {
    pub index: u32,
}

impl From<ParamId> for usize {
    fn from(id: ParamId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExprId {
    pub index: u32,
}

impl From<ExprId> for usize {
    fn from(id: ExprId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    Unit,
    Name { name: TokenId },
    Prod { fst: TypeId, snd: TypeId },
    Sum { left: TypeId, right: TypeId },
    Array { index: Option<TypeId>, elem: TypeId },
}

#[derive(Clone, Copy, Debug)]
pub enum Bind {
    Unit,
    Name {
        name: TokenId,
    },
    Pair {
        fst: ParamId,
        snd: ParamId,
    },
    Record {
        name: TokenId,
        field: ParamId,
        rest: ParamId,
    },
    End,
}

#[derive(Clone, Copy, Debug)]
pub struct Param {
    pub bind: Bind,
    pub ty: Option<TypeId>,
}

#[derive(Clone, Copy, Debug)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug)]
pub enum Expr {
    Name {
        name: TokenId,
    },
    Unit,
    Number {
        val: TokenId,
    },
    Pair {
        fst: ExprId,
        snd: ExprId,
    },
    Elem {
        array: ExprId,
        index: ExprId,
    },
    Apply {
        func: ExprId,
        arg: ExprId,
    },
    Map {
        func: ExprId,
        arg: ExprId,
    },
    Let {
        param: ParamId,
        val: ExprId,
        body: ExprId,
    },
    Binary {
        lhs: ExprId,
        op: Binop,
        rhs: ExprId,
    },
    Lambda {
        param: ParamId,
        ty: Option<TypeId>,
        body: ExprId,
    },
}

#[derive(Debug)]
pub struct Import {
    pub module: TokenId,
    pub names: Vec<TokenId>,
}

#[derive(Debug)]
pub struct Def {
    pub name: TokenId,
    pub types: Vec<TokenId>,
    pub params: Vec<ParamId>,
    pub ty: Option<TypeId>,
    pub body: ExprId,
}

#[derive(Debug)]
pub struct Module {
    imports: Vec<Import>,
    types: Vec<Type>,
    params: Vec<Param>,
    exprs: Vec<Expr>,
    defs: Vec<Def>,
}

impl Module {
    fn make_ty(&mut self, ty: Type) -> TypeId {
        let id = TypeId {
            index: self.types.len().try_into().unwrap(),
        };
        self.types.push(ty);
        id
    }

    fn make_param(&mut self, param: Param) -> ParamId {
        let id = ParamId {
            index: self.params.len().try_into().unwrap(),
        };
        self.params.push(param);
        id
    }

    fn make_expr(&mut self, expr: Expr) -> ExprId {
        let id = ExprId {
            index: self.exprs.len().try_into().unwrap(),
        };
        self.exprs.push(expr);
        id
    }

    pub fn ty(&self, id: TypeId) -> Type {
        self.types[usize::from(id)]
    }

    pub fn param(&self, id: ParamId) -> Param {
        self.params[usize::from(id)]
    }

    pub fn expr(&self, id: ExprId) -> Expr {
        self.exprs[usize::from(id)]
    }

    pub fn defs(&self) -> &[Def] {
        &self.defs
    }
}

#[derive(Debug)]
pub enum ParseError {
    Expected {
        id: TokenId,
        kinds: EnumSet<TokenKind>,
    },
}

#[derive(Debug)]
struct Parser<'a> {
    tokens: &'a Tokens,
    before_ws: TokenId,
    id: TokenId,
    module: Module,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> TokenKind {
        self.tokens.get(self.id).kind
    }

    fn find_non_ws(&mut self) {
        while let Newline | Comment = self.peek() {
            self.id.index += 1;
        }
    }

    fn next(&mut self) {
        if let Eof = self.peek() {
            panic!("unexpected end of file");
        }
        self.before_ws = TokenId {
            index: self.id.index + 1,
        };
        self.id = self.before_ws;
        self.find_non_ws();
    }

    fn expect(&mut self, kind: TokenKind) -> Result<TokenId, ParseError> {
        let id = self.id;
        if self.peek() == kind {
            self.next();
            Ok(id)
        } else {
            Err(ParseError::Expected {
                id,
                kinds: EnumSet::only(kind),
            })
        }
    }

    fn newline(&self) -> bool {
        // we only allow single-line comments, so anything ignored must include a newline
        self.before_ws < self.id
    }

    fn ty_atom(&mut self) -> Result<TypeId, ParseError> {
        match self.peek() {
            Ident => {
                let name = self.id;
                self.next();
                Ok(self.module.make_ty(Type::Name { name }))
            }
            LParen => {
                self.next();
                match self.peek() {
                    RParen => {
                        self.next();
                        Ok(self.module.make_ty(Type::Unit))
                    }
                    _ => {
                        let ty = self.ty()?;
                        self.expect(RParen)?;
                        Ok(ty)
                    }
                }
            }
            _ => Err(ParseError::Expected {
                id: self.id,
                kinds: Ident | LParen,
            }),
        }
    }

    fn ty_factor(&mut self) -> Result<TypeId, ParseError> {
        match self.peek() {
            LBracket => {
                self.next();
                let index = match self.peek() {
                    RBracket => None,
                    _ => Some(self.ty()?),
                };
                self.expect(RBracket)?;
                let elem = self.ty_factor()?;
                Ok(self.module.make_ty(Type::Array { index, elem }))
            }
            _ => self.ty_atom(),
        }
    }

    fn ty_term(&mut self) -> Result<TypeId, ParseError> {
        let mut types = vec![self.ty_factor()?];
        while let Asterisk = self.peek() {
            self.next();
            types.push(self.ty_factor()?);
        }
        let last = types.pop().unwrap();
        Ok(types.into_iter().rfold(last, |snd, fst| {
            self.module.make_ty(Type::Prod { fst, snd })
        }))
    }

    fn ty(&mut self) -> Result<TypeId, ParseError> {
        let mut types = vec![self.ty_term()?];
        while let Plus = self.peek() {
            self.next();
            types.push(self.ty_term()?);
        }
        let last = types.pop().unwrap();
        Ok(types.into_iter().rfold(last, |right, left| {
            self.module.make_ty(Type::Sum { left, right })
        }))
    }

    fn bind_atom(&mut self) -> Result<Bind, ParseError> {
        match self.peek() {
            Ident => {
                let name = self.id;
                self.next();
                Ok(Bind::Name { name })
            }
            LParen => {
                self.next();
                match self.peek() {
                    RParen => {
                        self.next();
                        Ok(Bind::Unit)
                    }
                    _ => {
                        let Param { bind, ty } = self.param()?;
                        let right = self.expect(RParen)?;
                        match ty {
                            Some(_) => Err(ParseError::Expected {
                                id: right,
                                kinds: EnumSet::only(Comma),
                            }),
                            None => Ok(bind),
                        }
                    }
                }
            }
            LBrace => {
                self.next();
                let mut fields = vec![];
                while let Ident = self.peek() {
                    let name = self.id;
                    self.next();
                    let bind = if let Equal = self.peek() {
                        self.next();
                        self.bind_elem()?
                    } else {
                        Bind::Name { name }
                    };
                    let ty = if let Colon = self.peek() {
                        self.next();
                        Some(self.ty()?)
                    } else {
                        None
                    };
                    fields.push((name, self.module.make_param(Param { bind, ty })));
                    match self.peek() {
                        Comma => self.next(),
                        _ => break,
                    }
                }
                self.expect(RBrace)?;
                Ok(fields
                    .into_iter()
                    .rfold(Bind::End, |bind, (name, field)| Bind::Record {
                        name,
                        field,
                        rest: self.module.make_param(Param { bind, ty: None }),
                    }))
            }
            _ => Err(ParseError::Expected {
                id: self.id,
                kinds: Ident | LParen,
            }),
        }
    }

    fn bind_elem(&mut self) -> Result<Bind, ParseError> {
        self.bind_atom()
    }

    fn param_elem(&mut self) -> Result<Param, ParseError> {
        let bind = self.bind_elem()?;
        let ty = match self.peek() {
            Colon => {
                self.next();
                Some(self.ty()?)
            }
            _ => None,
        };
        Ok(Param { bind, ty })
    }

    fn param(&mut self) -> Result<Param, ParseError> {
        let mut params = vec![self.param_elem()?];
        while let Comma = self.peek() {
            self.next();
            params.push(self.param_elem()?);
        }
        let last = params.pop().unwrap();
        Ok(params.into_iter().rfold(last, |snd, fst| Param {
            bind: Bind::Pair {
                fst: self.module.make_param(fst),
                snd: self.module.make_param(snd),
            },
            ty: None,
        }))
    }

    fn expr_atom(&mut self) -> Result<ExprId, ParseError> {
        match self.peek() {
            LParen => {
                self.next();
                match self.peek() {
                    RParen => {
                        self.next();
                        Ok(self.module.make_expr(Expr::Unit))
                    }
                    _ => {
                        let expr = self.expr()?;
                        self.expect(RParen)?;
                        Ok(expr)
                    }
                }
            }
            Ident => {
                let name = self.id;
                self.next();
                match self.peek() {
                    Arrow => {
                        self.next();
                        let bind = Bind::Name { name };
                        let param = self.module.make_param(Param { bind, ty: None });
                        let ty = None;
                        let body = self.expr()?;
                        Ok(self.module.make_expr(Expr::Lambda { param, ty, body }))
                    }
                    _ => Ok(self.module.make_expr(Expr::Name { name })),
                }
            }
            Number => {
                let val = self.id;
                self.next();
                Ok(self.module.make_expr(Expr::Number { val }))
            }
            _ => Err(ParseError::Expected {
                id: self.id,
                kinds: LParen | Ident | Number,
            }),
        }
    }

    fn expr_access(&mut self) -> Result<ExprId, ParseError> {
        let expr = self.expr_atom()?;
        match self.peek() {
            LBracket => {
                self.next();
                let index = self.expr()?;
                self.expect(RBracket)?;
                Ok(self.module.make_expr(Expr::Elem { array: expr, index }))
            }
            Dot => {
                self.next();
                self.expect(LParen)?;
                let arg = self.expr()?;
                self.expect(RParen)?;
                Ok(self.module.make_expr(Expr::Map { func: expr, arg }))
            }
            _ => Ok(expr),
        }
    }

    fn expr_factor(&mut self) -> Result<ExprId, ParseError> {
        let mut f = self.expr_access()?;
        // function application is the only place we forbid line breaks
        while !self.newline() {
            match self.peek() {
                // same set of tokens allowed at the start of an atomic expression
                LParen | Ident | Number => {
                    let x = self.expr_access()?;
                    f = self.module.make_expr(Expr::Apply { func: f, arg: x });
                }
                _ => break,
            }
        }
        Ok(f)
    }

    fn expr_term(&mut self) -> Result<ExprId, ParseError> {
        let mut lhs = self.expr_factor()?;
        loop {
            let op = match self.peek() {
                Asterisk => Binop::Mul,
                Slash => Binop::Div,
                _ => break,
            };
            self.next();
            let rhs = self.expr_factor()?;
            lhs = self.module.make_expr(Expr::Binary { lhs, op, rhs });
        }
        Ok(lhs)
    }

    fn expr_elem(&mut self) -> Result<ExprId, ParseError> {
        let mut lhs = self.expr_term()?;
        loop {
            let op = match self.peek() {
                Plus => Binop::Add,
                Hyphen => Binop::Sub,
                _ => break,
            };
            self.next();
            let rhs = self.expr_term()?;
            lhs = self.module.make_expr(Expr::Binary { lhs, op, rhs });
        }
        Ok(lhs)
    }

    fn expr_inner(&mut self) -> Result<ExprId, ParseError> {
        let mut exprs = vec![self.expr_elem()?];
        while let Comma = self.peek() {
            self.next();
            exprs.push(self.expr_elem()?);
        }
        let last = exprs.pop().unwrap();
        Ok(exprs.into_iter().rfold(last, |snd, fst| {
            self.module.make_expr(Expr::Pair { fst, snd })
        }))
    }

    fn expr(&mut self) -> Result<ExprId, ParseError> {
        match self.peek() {
            Let => {
                self.next();
                let param = self.param()?;
                let param = self.module.make_param(param);
                self.expect(Equal)?;
                let val = self.expr_inner()?;
                if !self.newline() {
                    self.expect(Semicolon)?;
                }
                let body = self.expr()?;
                Ok(self.module.make_expr(Expr::Let { param, val, body }))
            }
            _ => self.expr_inner(),
        }
    }

    fn import(&mut self) -> Result<Import, ParseError> {
        self.expect(Import)?;
        let module = self.expect(String)?;
        self.expect(Use)?;
        let mut names = vec![];
        while let Ident = self.peek() {
            names.push(self.id);
            self.next();
            match self.peek() {
                Comma => self.next(),
                _ => break,
            }
        }
        Ok(Import { module, names })
    }

    fn def(&mut self) -> Result<Def, ParseError> {
        self.expect(Def)?;
        let name = self.expect(Ident)?;
        let mut types = vec![];
        if let LBrace = self.peek() {
            self.next();
            while let Ident = self.peek() {
                types.push(self.id);
                self.next();
                match self.peek() {
                    Comma => self.next(),
                    _ => break,
                }
            }
            self.expect(RBrace)?;
        }
        let mut params = vec![];
        while let LParen = self.peek() {
            self.next();
            match self.peek() {
                RParen => {
                    self.next();
                    let bind = Bind::Unit;
                    params.push(self.module.make_param(Param { bind, ty: None }));
                }
                _ => {
                    let param = self.param()?;
                    params.push(self.module.make_param(param));
                    self.expect(RParen)?;
                }
            }
        }
        let ty = match self.peek() {
            Colon => {
                self.next();
                Some(self.ty()?)
            }
            _ => None,
        };
        self.expect(Equal)?;
        let body = self.expr()?;
        Ok(Def {
            name,
            types,
            params,
            ty,
            body,
        })
    }

    fn module(mut self) -> Result<Module, ParseError> {
        loop {
            match self.peek() {
                Import => {
                    let import = self.import()?;
                    self.module.imports.push(import);
                }
                Def => {
                    let def = self.def()?;
                    self.module.defs.push(def);
                }
                Eof => return Ok(self.module),
                _ => {
                    return Err(ParseError::Expected {
                        id: self.id,
                        kinds: Import | Def | Eof,
                    })
                }
            }
        }
    }
}

pub fn parse(tokens: &Tokens) -> Result<Module, ParseError> {
    let id = TokenId { index: 0 };
    let mut parser = Parser {
        tokens,
        before_ws: id,
        id,
        module: Module {
            imports: vec![],
            types: vec![],
            params: vec![],
            exprs: vec![],
            defs: vec![],
        },
    };
    parser.find_non_ws();
    parser.module()
}
