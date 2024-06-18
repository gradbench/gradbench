use std::fmt;

use chumsky::{
    input::{Stream, ValueInput},
    prelude::*,
};
use logos::Logos;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Logos, PartialEq)]
#[logos(skip r"(\s|#[^\n]*)+")]
pub enum Token<'input> {
    Error,

    #[regex(r"\w+")]
    Ident(&'input str),

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
}

impl<'a> fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Error => write!(f, "<error>"),
            Self::Ident(s) => write!(f, "{s}"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBracket => write!(f, "["),
            Self::RBracket => write!(f, "]"),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::Colon => write!(f, ":"),
            Self::Equal => write!(f, "="),
            Self::Plus => write!(f, "+"),
            Self::Hyphen => write!(f, "-"),
            Self::Asterisk => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Def => write!(f, "def"),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Param {
    name: String,
    ty: String,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Serialize)]
pub enum Expr {
    Var(String),
    Apply(Box<Expr>, Box<Expr>),
    Binary(Box<Expr>, Binop, Box<Expr>),
}

#[derive(Debug, Serialize)]
pub struct Def {
    name: String,
    params: Vec<Param>,
    body: Expr,
}

pub fn expr<'a, I>() -> impl Parser<'a, I, Expr, extra::Err<Rich<'a, Token<'a>>>>
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    recursive(|expr| {
        let var = select! {
            Token::Ident(s) => Expr::Var(s.to_string()),
        };

        let atom = var.or(expr.delimited_by(just(Token::LParen), just(Token::RParen)));

        let factor = atom.clone().foldl(atom.repeated(), |f, x| {
            Expr::Apply(Box::new(f), Box::new(x))
        });

        let term = factor.clone().foldl(
            just(Token::Asterisk)
                .to(Binop::Mul)
                .or(just(Token::Slash).to(Binop::Div))
                .then(factor)
                .repeated(),
            |x, (op, y)| Expr::Binary(Box::new(x), op, Box::new(y)),
        );

        term.clone().foldl(
            just(Token::Plus)
                .to(Binop::Add)
                .or(just(Token::Hyphen).to(Binop::Sub))
                .then(term)
                .repeated(),
            |x, (op, y)| Expr::Binary(Box::new(x), op, Box::new(y)),
        )
    })
}

pub fn parser<'a, I>() -> impl Parser<'a, I, Vec<Def>, extra::Err<Rich<'a, Token<'a>>>>
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    let name = select! {
        Token::Ident(s) => s.to_string(),
    };

    just(Token::Def)
        .ignore_then(name)
        .then(
            name.then_ignore(just(Token::Colon))
                .then(name)
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .map(|(name, ty)| Param { name, ty })
                .repeated()
                .collect(),
        )
        .then_ignore(just(Token::Equal))
        .then(expr())
        .map(|((name, params), body)| Def { name, params, body })
        .repeated()
        .collect()
        .then_ignore(end())
}

pub fn parse(input: &str) -> ParseResult<Vec<Def>, Rich<'_, Token<'_>>> {
    let tokens = Stream::from_iter(Token::lexer(input).spanned().map(|(tok, span)| match tok {
        Ok(tok) => (tok, span.into()),
        Err(()) => (Token::Error, span.into()),
    }))
    .spanned((input.len()..input.len()).into());
    parser().parse(tokens)
}
