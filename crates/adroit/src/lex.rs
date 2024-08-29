use std::{fmt, ops::Range};

use enumset::EnumSetType;
use logos::Logos;
use serde::Serialize;

use crate::util::{u32_to_usize, Id};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ByteIndex {
    pub index: u32,
}

impl Id for ByteIndex {
    fn from_usize(n: usize) -> Option<Self> {
        match n.try_into() {
            Ok(index) => Some(Self { index }),
            Err(_) => None,
        }
    }

    fn to_usize(self) -> usize {
        u32_to_usize(self.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ByteLen {
    pub len: u16,
}

impl Id for ByteLen {
    fn from_usize(n: usize) -> Option<Self> {
        match n.try_into() {
            Ok(len) => Some(Self { len }),
            Err(_) => None,
        }
    }

    fn to_usize(self) -> usize {
        self.len.into()
    }
}

#[derive(Debug, EnumSetType, Hash, Logos, Serialize)]
#[logos(skip r"[^\S\r\n]+")]
pub enum TokenKind {
    Eof,

    #[regex("\r?\n")]
    Newline,

    #[regex("#[^\n]*")]
    Comment,

    #[regex(r"[A-Z_a-z]\w*")]
    Ident,

    #[regex(r"\d+(\.\d+)?")]
    Number,

    #[regex(r#""[^"]*""#)]
    String,

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

    #[token(",")]
    Comma,

    #[token(".")]
    Dot,

    #[token(":")]
    Colon,

    #[token("=")]
    Equal,

    #[token(";")]
    Semicolon,

    #[token("+")]
    Plus,

    #[token("-")]
    Dash,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token(".*")]
    DotStar,

    #[token("./")]
    DotSlash,

    #[token("->")]
    To,

    #[token("=>")]
    Arrow,

    #[token("<-")]
    Gets,

    #[token("def")]
    Def,

    #[token("import")]
    Import,

    #[token("index")]
    Index,

    #[token("let")]
    Let,

    #[token("undefined")]
    Undefined,

    #[token("use")]
    Use,
}

impl TokenKind {
    pub fn ignore(self) -> bool {
        matches!(self, Self::Newline | Self::Comment)
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Eof => write!(f, "end of file"),
            Self::Newline => write!(f, "newline"),
            Self::Comment => write!(f, "comment"),
            Self::Ident => write!(f, "identifier"),
            Self::Number => write!(f, "number"),
            Self::String => write!(f, "string"),
            Self::LParen => write!(f, "`(`"),
            Self::RParen => write!(f, "`)`"),
            Self::LBracket => write!(f, "`[`"),
            Self::RBracket => write!(f, "`]`"),
            Self::LBrace => write!(f, "`{{`"),
            Self::RBrace => write!(f, "`}}`"),
            Self::Comma => write!(f, "`,`"),
            Self::Dot => write!(f, "`.`"),
            Self::Colon => write!(f, "`:`"),
            Self::Equal => write!(f, "`=`"),
            Self::Semicolon => write!(f, "`;`"),
            Self::Plus => write!(f, "`+`"),
            Self::Dash => write!(f, "`-`"),
            Self::Star => write!(f, "`*`"),
            Self::Slash => write!(f, "`/`"),
            Self::DotStar => write!(f, "`.*`"),
            Self::DotSlash => write!(f, "`./`"),
            Self::To => write!(f, "`->`"),
            Self::Arrow => write!(f, "`=>`"),
            Self::Gets => write!(f, "`<-`"),
            Self::Def => write!(f, "`def`"),
            Self::Import => write!(f, "`import`"),
            Self::Index => write!(f, "`index`"),
            Self::Let => write!(f, "`let`"),
            Self::Undefined => write!(f, "`undefined`"),
            Self::Use => write!(f, "`use`"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Token {
    pub start: ByteIndex,
    pub len: ByteLen,
    pub kind: TokenKind,
}

impl Token {
    pub fn byte_range(&self) -> Range<usize> {
        let start = self.start.to_usize();
        start..(start + self.len.to_usize())
    }

    pub fn string(&self, source: &str) -> String {
        let kind = self.kind;
        assert_eq!(kind, TokenKind::String, "the {kind} token is not a string");
        serde_json::from_str(&source[self.byte_range()]).expect("strings should be valid JSON")
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct TokenId {
    pub index: u32,
}

impl Id for TokenId {
    fn from_usize(n: usize) -> Option<Self> {
        match n.try_into() {
            Ok(index) => Some(Self { index }),
            Err(_) => None,
        }
    }

    fn to_usize(self) -> usize {
        u32_to_usize(self.index)
    }
}

#[derive(Debug, Serialize)]
#[serde(transparent)]
pub struct Tokens {
    tokens: Vec<Token>,
}

impl Tokens {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn get(&self, index: TokenId) -> Token {
        self.tokens[index.to_usize()]
    }
}

#[derive(Debug)]
pub enum LexError {
    SourceTooLong,
    TokenTooLong { start: ByteIndex, end: ByteIndex },
    InvalidToken { start: ByteIndex, len: ByteLen },
}

impl LexError {
    pub fn byte_range(&self) -> Range<usize> {
        match *self {
            LexError::SourceTooLong => {
                let max = ByteIndex { index: u32::MAX }.to_usize();
                max..max
            }
            LexError::TokenTooLong { start, end } => start.to_usize()..end.to_usize(),
            LexError::InvalidToken { start, len } => {
                let start = start.to_usize();
                start..(start + len.to_usize())
            }
        }
    }

    pub fn message(&self) -> &str {
        match self {
            LexError::SourceTooLong { .. } => "file size exceeds 4 GiB limit",
            LexError::TokenTooLong { .. } => "token size exceeds 64 KiB limit",
            LexError::InvalidToken { .. } => "invalid token",
        }
    }
}

pub fn lex(source: &str) -> Result<Tokens, LexError> {
    let eof = match u32::try_from(source.len()) {
        Ok(len) => Token {
            start: ByteIndex { index: len },
            len: ByteLen { len: 0 },
            kind: TokenKind::Eof,
        },
        Err(_) => return Err(LexError::SourceTooLong),
    };
    let mut tokens = Vec::new();
    for (result, range) in TokenKind::lexer(source).spanned() {
        let start = ByteIndex::from_usize(range.start)
            .expect("file size limit should ensure all token starts are in range");
        let end = ByteIndex::from_usize(range.end)
            .expect("file size limit should ensure all token ends are in range");
        let len = ByteLen {
            len: (end.index - start.index)
                .try_into()
                .map_err(|_| LexError::TokenTooLong { start, end })?,
        };
        let kind = result.map_err(|_| LexError::InvalidToken { start, len })?;
        tokens.push(Token { start, len, kind });
    }
    tokens.push(eof);
    Ok(Tokens { tokens })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crlf() {
        let actual: Vec<TokenKind> = lex("a\r\nb")
            .unwrap()
            .tokens
            .into_iter()
            .map(|tok| tok.kind)
            .collect();
        let expected = vec![
            TokenKind::Ident,
            TokenKind::Newline,
            TokenKind::Ident,
            TokenKind::Eof,
        ];
        assert_eq!(actual, expected);
    }
}
