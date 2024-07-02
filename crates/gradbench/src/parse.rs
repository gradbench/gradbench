use crate::lex::{Token, TokenId, TokenKind, Tokens};

#[derive(Debug)]
pub struct ExprId {
    pub index: u32,
}

#[derive(Debug)]
pub struct Ident {
    pub name: TokenId,
}

#[derive(Debug)]
pub struct Param {
    pub name: Ident,
    pub ty: Ident,
}

#[derive(Clone, Copy, Debug)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug)]
pub enum Expr {
    Var {
        name: Ident,
    },
    Apply {
        func: ExprId,
        arg: ExprId,
    },
    Let {
        var: Ident,
        def: ExprId,
        body: ExprId,
    },
    Binary {
        lhs: ExprId,
        op: Binop,
        rhs: ExprId,
    },
}

#[derive(Debug)]
pub struct Def {
    pub name: TokenId,
    pub params: Vec<Param>,
    pub body: ExprId,
}

#[derive(Debug)]
pub struct Module {
    pub exprs: Vec<Expr>,
    pub defs: Vec<Def>,
}

impl Module {
    pub fn expr(&mut self, expr: Expr) -> ExprId {
        let id = ExprId {
            index: self.exprs.len().try_into().unwrap(),
        };
        self.exprs.push(expr);
        id
    }
}

#[derive(Debug)]
pub enum ParseError {
    Expected { id: TokenId, kind: TokenKind },
    UnexpectedToplevel { id: TokenId },
    ExpectedExpression { id: TokenId },
    ExpectedStatementEnd { id: TokenId },
}

use ParseError::*;
use TokenKind::*;

struct Parser<'a> {
    tokens: &'a Tokens,
    id: TokenId,
    module: Module,
}

impl<'a> Parser<'a> {
    fn current(&self) -> Token {
        self.tokens.get(self.id)
    }

    fn advance(&mut self) {
        self.id = TokenId {
            index: self.id.index + 1,
        };
    }

    fn ignore(&mut self) {
        while let Newline = self.current().kind {
            self.advance();
        }
    }

    fn validate(
        &mut self,
        pred: impl FnOnce(TokenKind) -> bool,
        err: impl FnOnce(TokenId, TokenKind) -> ParseError,
    ) -> Result<TokenId, ParseError> {
        let id = self.id;
        let kind = self.current().kind;
        if pred(kind) {
            self.advance();
            Ok(id)
        } else {
            Err(err(id, kind))
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<TokenId, ParseError> {
        self.ignore();
        self.validate(|k| k == kind, |id, _| Expected { id, kind })
    }

    fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        let name = self.expect(Ident)?;
        Ok(Ident { name })
    }

    fn parse_expr_atom(&mut self) -> Result<ExprId, ParseError> {
        match self.current().kind {
            LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(RParen)?;
                Ok(expr)
            }
            Ident => {
                let name = self.parse_ident()?;
                Ok(self.module.expr(Expr::Var { name }))
            }
            _ => Err(ExpectedExpression { id: self.id }),
        }
    }

    fn parse_expr_factor(&mut self) -> Result<ExprId, ParseError> {
        self.ignore();
        let mut f = self.parse_expr_atom()?;
        // function application is the only place we forbid line breaks
        while let LParen | Ident = self.current().kind {
            let x = self.parse_expr_atom()?;
            f = self.module.expr(Expr::Apply { func: f, arg: x });
        }
        Ok(f)
    }

    fn parse_expr_term(&mut self) -> Result<ExprId, ParseError> {
        let mut lhs = self.parse_expr_factor()?;
        loop {
            self.ignore();
            let op = match self.current().kind {
                Asterisk => Binop::Mul,
                Slash => Binop::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_expr_factor()?;
            lhs = self.module.expr(Expr::Binary { lhs, op, rhs });
        }
        Ok(lhs)
    }

    fn parse_expr_inner(&mut self) -> Result<ExprId, ParseError> {
        let mut lhs = self.parse_expr_term()?;
        loop {
            self.ignore();
            let op = match self.current().kind {
                Plus => Binop::Add,
                Hyphen => Binop::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_expr_term()?;
            lhs = self.module.expr(Expr::Binary { lhs, op, rhs });
        }
        Ok(lhs)
    }

    fn parse_expr(&mut self) -> Result<ExprId, ParseError> {
        self.ignore();
        match self.current().kind {
            Let => {
                self.advance();
                let var = self.parse_ident()?;
                self.expect(Equal)?;
                let def = self.parse_expr_inner()?;
                self.validate(
                    |kind| kind == Newline || kind == Semicolon,
                    |id, _| ExpectedStatementEnd { id },
                )?;
                let body = self.parse_expr()?;
                Ok(self.module.expr(Expr::Let { var, def, body }))
            }
            _ => self.parse_expr_inner(),
        }
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let name = self.parse_ident()?;
        self.expect(Colon)?;
        let ty = self.parse_ident()?;
        Ok(Param { name, ty })
    }

    fn parse_def(&mut self) -> Result<Def, ParseError> {
        self.expect(Def)?;
        let name = self.expect(Ident)?;
        let mut params = vec![];
        self.ignore();
        if let LParen = self.current().kind {
            self.advance();
            loop {
                self.ignore();
                match self.current().kind {
                    RParen => {
                        self.advance();
                        break;
                    }
                    _ => {
                        params.push(self.parse_param()?);
                        self.ignore();
                        match self.current().kind {
                            Comma => self.advance(),
                            RParen => {
                                self.advance();
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        self.expect(Equal)?;
        let body = self.parse_expr()?;
        Ok(Def { name, params, body })
    }

    fn parse_module(mut self) -> Result<Module, ParseError> {
        loop {
            self.ignore();
            match self.current().kind {
                Def => {
                    let def = self.parse_def()?;
                    self.module.defs.push(def);
                }
                Eof => break,
                _ => return Err(UnexpectedToplevel { id: self.id }),
            }
        }
        Ok(self.module)
    }
}

pub fn parse(tokens: &Tokens) -> Result<Module, ParseError> {
    Parser {
        tokens,
        id: TokenId { index: 0 },
        module: Module {
            exprs: vec![],
            defs: vec![],
        },
    }
    .parse_module()
}
