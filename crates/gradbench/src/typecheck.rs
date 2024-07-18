use std::collections::HashMap;

use indexmap::IndexSet;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Type {
    Var { module: Option<ModId>, def: TokenId },
    Unit,
    Func { dom: TypeId, cod: TypeId },
}

#[derive(Clone, Copy, Debug)]
enum Expr {
    Use { module: ModId, id: ValId },
    Lambda,
}

#[derive(Clone, Copy, Debug)]
struct Val {
    ty: TypeId,
    expr: Expr,
}

#[derive(Debug)]
pub struct Module {
    imports: Vec<String>,
    exports: HashMap<String, ValId>,
    types: IndexSet<Type>,
    vals: Vec<Val>,
}

impl Module {
    fn export(&self, name: &str) -> Option<ValId> {
        self.exports.get(name).copied()
    }

    fn val(&self, id: ValId) -> Val {
        self.vals[usize::from(id)]
    }
}

#[derive(Debug)]
pub enum TypeError {
    TooManyImports,
    TooManyTypes,
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

    fn ty(&mut self, ty: Type) -> Result<TypeId, TypeError> {
        let (i, _) = self.module.types.insert_full(ty);
        let id = TypeId {
            // maybe more types than tokens, because of imports
            index: i.try_into().map_err(|_| TypeError::TooManyTypes)?,
        };
        Ok(id)
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
            let mod_id = ModId { index };
            let module = self.imports[usize::from(mod_id)];
            let mut needed = vec![false; module.types.len()];
            let uses = imports[usize::from(mod_id)]
                .names
                .iter()
                .map(|&token| {
                    let name = self.token(token);
                    let id = module
                        .export(name)
                        .ok_or(TypeError::Undefined { name: token })?;
                    needed[usize::from(module.val(id).ty)] = true;
                    Ok((name, id))
                })
                .collect::<Result<Vec<(&'a str, ValId)>, TypeError>>()?;
            for (name, id) in uses {
                let ty = self.ty(Type::Unit)?;
                let val = self.val(Val {
                    ty,
                    expr: Expr::Use { module: mod_id, id },
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
            types: IndexSet::new(),
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
        types: IndexSet::from([
            Type::Unit,
            Type::Func {
                dom: TypeId { index: 0 },
                cod: TypeId { index: 0 },
            },
        ]),
        vals: vec![Val {
            ty: TypeId { index: 1 },
            expr: Expr::Lambda,
        }],
    }
}

pub fn math() -> Module {
    Module {
        imports: vec![],
        exports: HashMap::from(
            ["exp", "int", "lgamma", "log", "pi", "sqr", "sqrt"]
                .map(|s| (s.to_owned(), ValId { index: 0 })),
        ),
        types: IndexSet::from([
            Type::Unit,
            Type::Func {
                dom: TypeId { index: 0 },
                cod: TypeId { index: 0 },
            },
        ]),
        vals: vec![Val {
            ty: TypeId { index: 1 },
            expr: Expr::Lambda,
        }],
    }
}
