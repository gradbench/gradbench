use crate::lex::{TokenId, TokenKind, Tokens};

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
    Number {
        val: TokenId,
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
    ExpectedParamEnd { id: TokenId },
    ExpectedStatementEnd { id: TokenId },
    ExpectedExpression { id: TokenId },
}

use ParseError::*;
use TokenKind::*;

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

    fn validate(
        &mut self,
        pred: impl FnOnce(TokenKind) -> bool,
        err: impl FnOnce(TokenId, TokenKind) -> ParseError,
    ) -> Result<TokenId, ParseError> {
        let id = self.id;
        let kind = self.peek();
        if pred(kind) {
            self.next();
            Ok(id)
        } else {
            Err(err(id, kind))
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<TokenId, ParseError> {
        self.validate(|k| k == kind, |id, _| Expected { id, kind })
    }

    fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        let name = self.expect(Ident)?;
        Ok(Ident { name })
    }

    fn parse_expr_atom(&mut self) -> Result<ExprId, ParseError> {
        match self.peek() {
            LParen => {
                self.next();
                let expr = self.parse_expr()?;
                self.expect(RParen)?;
                Ok(expr)
            }
            Ident => {
                let name = self.parse_ident()?;
                Ok(self.module.expr(Expr::Var { name }))
            }
            Number => {
                self.next();
                Ok(self.module.expr(Expr::Number { val: self.id }))
            }
            _ => Err(ExpectedExpression { id: self.id }),
        }
    }

    fn parse_expr_factor(&mut self) -> Result<ExprId, ParseError> {
        let mut f = self.parse_expr_atom()?;
        // function application is the only place we forbid line breaks
        while let LParen | Ident = self.peek() {
            let x = self.parse_expr_atom()?;
            f = self.module.expr(Expr::Apply { func: f, arg: x });
        }
        Ok(f)
    }

    fn parse_expr_term(&mut self) -> Result<ExprId, ParseError> {
        let mut lhs = self.parse_expr_factor()?;
        loop {
            let op = match self.peek() {
                Asterisk => Binop::Mul,
                Slash => Binop::Div,
                _ => break,
            };
            self.next();
            let rhs = self.parse_expr_factor()?;
            lhs = self.module.expr(Expr::Binary { lhs, op, rhs });
        }
        Ok(lhs)
    }

    fn parse_expr_inner(&mut self) -> Result<ExprId, ParseError> {
        let mut lhs = self.parse_expr_term()?;
        loop {
            let op = match self.peek() {
                Plus => Binop::Add,
                Hyphen => Binop::Sub,
                _ => break,
            };
            self.next();
            let rhs = self.parse_expr_term()?;
            lhs = self.module.expr(Expr::Binary { lhs, op, rhs });
        }
        Ok(lhs)
    }

    fn parse_expr(&mut self) -> Result<ExprId, ParseError> {
        match self.peek() {
            Let => {
                self.next();
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
        if let LParen = self.peek() {
            self.next();
            loop {
                match self.peek() {
                    RParen => {
                        self.next();
                        break;
                    }
                    _ => {
                        params.push(self.parse_param()?);
                        match self.peek() {
                            Comma => self.next(),
                            RParen => {
                                self.next();
                                break;
                            }
                            _ => return Err(ExpectedParamEnd { id: self.id }),
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
            match self.peek() {
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
    let id = TokenId { index: 0 };
    let mut parser = Parser {
        tokens,
        before_ws: id,
        id,
        module: Module {
            exprs: vec![],
            defs: vec![],
        },
    };
    parser.find_non_ws();
    parser.parse_module()
}

#[cfg(test)]
mod tests {
    use crate::lex;

    use super::*;

    #[test]
    fn test_empty() {
        let Module { exprs, defs } = parse(&lex::lex("").unwrap()).unwrap();
        assert!(exprs.is_empty());
        assert!(defs.is_empty());
    }

    #[test]
    fn test_leading_comment() {
        let result = parse(&lex::lex(include_str!("leading_comment.adroit")).unwrap());
        assert!(result.is_ok());
    }
}
