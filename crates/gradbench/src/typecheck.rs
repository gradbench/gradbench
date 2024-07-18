use std::collections::HashMap;

use crate::{
    lex::{TokenId, Tokens},
    parse,
    util::u32_to_usize,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModId {
    pub index: u16,
}

impl From<ModId> for usize {
    fn from(id: ModId) -> Self {
        id.index.into()
    }
}

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
pub struct ValId {
    pub index: u32,
}

impl From<ValId> for usize {
    fn from(id: ValId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Debug)]
enum Type {
    Var { def: TokenId },
}

#[derive(Debug)]
enum Val {
    Use { module: ModId, name: ValId },
}

#[derive(Debug)]
pub struct Module {
    pub imports: Vec<String>,
    pub exports: HashMap<String, ValId>,
    pub types: Vec<Type>,
    pub vals: Vec<Val>,
}

#[derive(Debug)]
pub enum TypeError {
    TooManyImports,
    Undefined { name: TokenId },
}

#[derive(Debug)]
struct Typer<'a> {
    imports: Vec<&'a Module>,
    source: &'a str,
    tokens: &'a Tokens,
    tree: &'a parse::Module,
    module: Module,
    names: HashMap<&'a str, Vec<ValId>>,
}

impl<'a> Typer<'a> {
    fn token(&self, id: TokenId) -> &'a str {
        &self.source[self.tokens.get(id).byte_range()]
    }

    fn val(&mut self, val: Val) -> ValId {
        let id = ValId {
            // we assume there are at least as many tokens as values
            index: self.module.vals.len().try_into().unwrap(),
        };
        self.module.vals.push(val);
        id
    }

    fn imports(&mut self) -> Result<(), TypeError> {
        let n = self.tree.imports().len();
        if n == 0 {
            return Ok(()); // avoid underflow when we decrement below
        }
        let imports = self.tree.imports();
        for index in 0..=(n - 1).try_into().map_err(|_| TypeError::TooManyImports)? {
            let module = ModId { index };
            let exports = &self.imports[usize::from(module)].exports;
            for &token in &imports[usize::from(module)].names {
                let name = self.token(token);
                let val = self.val(Val::Use {
                    module,
                    name: *exports
                        .get(name)
                        .ok_or(TypeError::Undefined { name: token })?,
                });
                self.names.insert(name, vec![val]);
            }
        }
        Ok(())
    }

    fn module(mut self) -> Result<Module, TypeError> {
        self.imports()?;
        Ok(self.module)
    }
}

pub fn typecheck<'a>(
    mut import: impl FnMut(&str) -> &'a Module,
    source: &'a str,
    tokens: &'a Tokens,
    tree: &'a parse::Module,
) -> Result<Module, TypeError> {
    let imports: Vec<String> = tree
        .imports()
        .iter()
        .map(|imp| tokens.get(imp.module).string(source))
        .collect();
    Typer {
        imports: tree
            .imports()
            .iter()
            .enumerate()
            .map(|(i, _)| import(&imports[i]))
            .collect(),
        source,
        tokens,
        tree,
        module: Module {
            imports,
            exports: HashMap::new(),
            types: vec![],
            vals: vec![],
        },
        names: HashMap::new(),
    }
    .module()
}

pub fn array() -> Module {
    Module {
        imports: vec![],
        exports: HashMap::from(
            [
                "array",
                "concat",
                "for",
                "map",
                "max",
                "range",
                "row",
                "scan",
                "slice",
                "stack",
                "sum",
                "transpose",
                "zeros",
            ]
            .map(|s| (s.to_owned(), ValId { index: 0 })),
        ),
        types: vec![],
        vals: vec![],
    }
}

pub fn math() -> Module {
    Module {
        imports: vec![],
        exports: HashMap::from(
            ["exp", "int", "lgamma", "log", "pi", "sqr", "sqrt"]
                .map(|s| (s.to_owned(), ValId { index: 0 })),
        ),
        types: vec![],
        vals: vec![],
    }
}
