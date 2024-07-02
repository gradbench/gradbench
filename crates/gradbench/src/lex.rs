use std::ops::Range;

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

#[derive(Debug, Logos)]
#[logos(skip r"[^\S\r\n]+(#[^\n]*)?")]
pub enum TokenKind {
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

#[derive(Debug)]
pub struct Token {
    pub start: ByteIndex,
    pub len: ByteLen,
    pub kind: TokenKind,
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

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    if u32::try_from(source.len()).is_err() {
        return Err(LexError::SourceTooLong);
    }
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
    Ok(tokens)
}
