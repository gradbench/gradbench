use std::fmt;

use chumsky::{
    input::{Stream, ValueInput},
    prelude::*,
};
use logos::Logos;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Logos, PartialEq)]
#[logos(skip r"[^\S\r\n]+(#[^\n]*)?")]
pub enum Token<'a> {
    Error,

    #[token("\n")]
    Newline,

    #[regex(r"\w+")]
    Ident(&'a str),

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token(":")]
    Colon,

    #[token("=")]
    Equal,

    #[token(";")]
    Semicolon,

    #[token("+")]
    Plus,

    #[token("-")]
    Hyphen,

    #[token("*")]
    Asterisk,

    #[token("/")]
    Slash,

    #[token("def")]
    Def,

    #[token("let")]
    Let,
}

impl<'a> fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Error => write!(f, "<error>"),
            Self::Newline => write!(f, "<newline>"),
            Self::Ident(s) => write!(f, "{s}"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBracket => write!(f, "["),
            Self::RBracket => write!(f, "]"),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::Colon => write!(f, ":"),
            Self::Equal => write!(f, "="),
            Self::Semicolon => write!(f, ";"),
            Self::Plus => write!(f, "+"),
            Self::Hyphen => write!(f, "-"),
            Self::Asterisk => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Def => write!(f, "def"),
            Self::Let => write!(f, "let"),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Param<'a> {
    name: &'a str,
    ty: &'a str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Serialize)]
pub enum Expr<'a> {
    Var {
        name: &'a str,
    },
    Apply {
        func: Box<Expr<'a>>,
        arg: Box<Expr<'a>>,
    },
    Let {
        var: &'a str,
        def: Box<Expr<'a>>,
        body: Box<Expr<'a>>,
    },
    Binary {
        lhs: Box<Expr<'a>>,
        op: Binop,
        rhs: Box<Expr<'a>>,
    },
}

#[derive(Debug, Serialize)]
pub struct Def<'a> {
    name: &'a str,
    params: Vec<Param<'a>>,
    body: Expr<'a>,
}

#[derive(Debug, Serialize)]
pub struct Module<'a> {
    defs: Vec<Def<'a>>,
}

pub fn parser<'a, I>() -> impl Parser<'a, I, Module<'a>, extra::Err<Rich<'a, Token<'a>>>>
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    let space = just(Token::Newline).repeated();

    let ident = any().try_map(|tok, span| match tok {
        Token::Ident(s) => Ok(s),
        _ => Err(Rich::custom(
            span,
            format!("found '{tok}', expected '<identifier>'"),
        )),
    });

    let expr = recursive(|expr| {
        let atom = ident
            .map(|name| Expr::Var { name })
            .or(expr.clone().delimited_by(
                just(Token::LParen).ignore_then(space),
                space.then_ignore(just(Token::RParen)),
            ));

        // function application is the only place we forbid line breaks
        let factor = atom.clone().foldl(atom.repeated(), |f, x| Expr::Apply {
            func: Box::new(f),
            arg: Box::new(x),
        });

        let term = factor.clone().foldl(
            space
                .ignore_then(
                    just(Token::Asterisk)
                        .to(Binop::Mul)
                        .or(just(Token::Slash).to(Binop::Div)),
                )
                .then_ignore(space)
                .then(factor)
                .repeated(),
            |x, (op, y)| Expr::Binary {
                lhs: Box::new(x),
                op,
                rhs: Box::new(y),
            },
        );

        let inner = term.clone().foldl(
            space
                .ignore_then(
                    just(Token::Plus)
                        .to(Binop::Add)
                        .or(just(Token::Hyphen).to(Binop::Sub)),
                )
                .then_ignore(space)
                .then(term)
                .repeated(),
            |x, (op, y)| Expr::Binary {
                lhs: Box::new(x),
                op,
                rhs: Box::new(y),
            },
        );

        let bind = just(Token::Let)
            .ignore_then(space)
            .ignore_then(ident)
            .then_ignore(space)
            .then_ignore(just(Token::Equal))
            .then_ignore(space)
            .then(inner.clone())
            .then_ignore(just(Token::Newline).or(just(Token::Semicolon)))
            .then_ignore(space)
            .then(expr)
            .map(|((var, def), body)| Expr::Let {
                var,
                def: Box::new(def),
                body: Box::new(body),
            });

        bind.or(inner)
    });

    let param = ident
        .then_ignore(space)
        .then_ignore(just(Token::Colon))
        .then_ignore(space)
        .then(ident)
        .delimited_by(
            just(Token::LParen).ignore_then(space),
            space.then_ignore(just(Token::RParen)),
        )
        .map(|(name, ty)| Param { name, ty });

    let def = just(Token::Def)
        .ignore_then(space)
        .ignore_then(ident)
        .then_ignore(space)
        .then(param.then_ignore(space).repeated().collect())
        .then_ignore(just(Token::Equal))
        .then_ignore(space)
        .then(expr)
        .map(|((name, params), body)| Def { name, params, body });

    space
        .ignore_then(def.then_ignore(space).repeated().collect())
        .then_ignore(end())
        .map(|defs| Module { defs })
}

pub fn parse(input: &str) -> ParseResult<Module, Rich<'_, Token<'_>>> {
    let tokens = Stream::from_iter(Token::lexer(input).spanned().map(|(tok, span)| match tok {
        Ok(tok) => (tok, span.into()),
        Err(()) => (Token::Error, span.into()),
    }))
    .spanned((input.len()..input.len()).into());
    parser().parse(tokens)
}
