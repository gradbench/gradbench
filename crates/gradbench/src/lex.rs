use std::{fmt, ops::Range};

use enumset::EnumSetType;
use logos::Logos;
use serde::Serialize;

use crate::util::u32_to_usize;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ByteIndex {
    pub index: u32,
}

impl From<ByteIndex> for usize {
    fn from(index: ByteIndex) -> Self {
        u32_to_usize(index.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ByteLen {
    pub len: u16,
}

impl From<ByteLen> for usize {
    fn from(len: ByteLen) -> Self {
        len.len.into()
    }
}

#[derive(Debug, EnumSetType, Hash, Logos)]
#[logos(skip r"[^\S\r\n]+")]
pub enum TokenKind {
    Eof,

    #[token("\n")]
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

    #[token("use")]
    Use,
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
            Self::Use => write!(f, "`use`"),
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

impl From<TokenId> for usize {
    fn from(id: TokenId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Debug)]
pub struct Tokens {
    tokens: Vec<Token>,
}

impl Tokens {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

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
            index: range
                .start
                .try_into()
                .expect("file size limit should ensure all token starts are in range"),
        };
        let end = ByteIndex {
            index: range
                .end
                .try_into()
                .expect("file size limit should ensure all token ends are in range"),
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
