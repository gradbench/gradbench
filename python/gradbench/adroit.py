from enum import Enum
from typing import NewType

from pydantic import BaseModel

ByteIndex = NewType("ByteIndex", int)
ByteLen = NewType("ByteLen", int)


class TokenKind(str, Enum):
    Eof = "Eof"
    Newline = "Newline"
    Comment = "Comment"
    Ident = "Ident"
    Number = "Number"
    String = "String"
    LParen = "LParen"
    RParen = "RParen"
    LBracket = "LBracket"
    RBracket = "RBracket"
    LBrace = "LBrace"
    Comma = "Comma"
    Dot = "Dot"
    Colon = "Colon"
    Equal = "Equal"
    Semicolon = "Semicolon"
    Plus = "Plus"
    Dash = "Dash"
    Star = "Star"
    Slash = "Slash"
    DotStar = "DotStar"
    DotSlash = "DotSlash"
    To = "To"
    Arrow = "Arrow"
    Gets = "Gets"
    Def = "Def"
    Import = "Import"
    Index = "Index"
    Let = "Let"
    Undefined = "Undefined"
    Use = "Use"


class Token(BaseModel):
    start: ByteIndex
    len: ByteLen
    kind: TokenKind


TokenId = NewType("TokenId", int)

ParseTypeId = NewType("ParseTypeId", int)
ParamId = NewType("ParamId", int)
ExprId = NewType("ExprId", int)
DefId = NewType("DefId", int)


class Def(BaseModel):
    name: TokenId


class ParseModule(BaseModel):
    imports: list[Import]
    types: list[ParseType]
    params: list[Param]
    exprs: list[Expr]
    defs: list[Def]


class TypecheckModule(BaseModel):
    pass


class FullNode(BaseModel):
    source: str
    tokens: list[Token]
    tree: ParseModule
    imports: list[str]
    module: TypecheckModule


class Modules(BaseModel):
    root: str
    modules: dict[str, FullNode]
