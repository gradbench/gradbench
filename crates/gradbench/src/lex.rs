use std::{fmt, ops::Range};

use logos::Logos;

#[derive(Clone, Copy, Debug)]
pub struct ByteIndex {
    pub index: u32,
}

impl From<ByteIndex> for usize {
    fn from(index: ByteIndex) -> Self {
        index
            .index
            .try_into()
            .expect("pointer size is assumed to be at least 32 bits")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ByteLen {
    pub len: u16,
}

impl From<ByteLen> for usize {
    fn from(len: ByteLen) -> Self {
        len.len.into()
    }
}

#[derive(Clone, Copy, Debug, Logos, PartialEq)]
#[logos(skip r"[^\S\r\n]+(#[^\n]*)?")]
pub enum TokenKind {
    Eof,

    #[token("\n")]
    Newline,

    #[regex(r"\w+")]
    Ident,

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

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Eof => write!(f, "end of file"),
            Self::Newline => write!(f, "newline"),
            Self::Ident => write!(f, "identifier"),
            Self::LParen => write!(f, "`(`"),
            Self::RParen => write!(f, "`)`"),
            Self::LBracket => write!(f, "`[`"),
            Self::RBracket => write!(f, "`]`"),
            Self::LBrace => write!(f, "`{{`"),
            Self::RBrace => write!(f, "`}}`"),
            Self::Comma => write!(f, "`,`"),
            Self::Colon => write!(f, "`:`"),
            Self::Equal => write!(f, "`=`"),
            Self::Semicolon => write!(f, "`;`"),
            Self::Plus => write!(f, "`+`"),
            Self::Hyphen => write!(f, "`-`"),
            Self::Asterisk => write!(f, "`*`"),
            Self::Slash => write!(f, "`/`"),
            Self::Def => write!(f, "`def`"),
            Self::Let => write!(f, "`let`"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub start: ByteIndex,
    pub len: ByteLen,
    pub kind: TokenKind,
}

impl Token {
    pub fn byte_range(&self) -> Range<usize> {
        let start = usize::from(self.start);
        start..(start + usize::from(self.len))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TokenId {
    pub index: u32,
}

impl From<TokenId> for usize {
    fn from(index: TokenId) -> Self {
        index
            .index
            .try_into()
            .expect("pointer size is assumed to be at least 32 bits")
    }
}

#[derive(Debug)]
pub struct Tokens {
    pub tokens: Vec<Token>,
}

impl Tokens {
    pub fn get(&self, index: TokenId) -> Token {
        self.tokens[usize::from(index)]
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
                let max = usize::from(ByteIndex { index: u32::MAX });
                max..max
            }
            LexError::TokenTooLong { start, end } => usize::from(start)..usize::from(end),
            LexError::InvalidToken { start, len } => {
                let start = usize::from(start);
                start..(start + usize::from(len))
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
        let start = ByteIndex {
            index: range.start.try_into().unwrap(),
        };
        let end = ByteIndex {
            index: range.end.try_into().unwrap(),
        };
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
