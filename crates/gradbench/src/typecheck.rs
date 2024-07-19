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
    Int,
    Float,
    Prod { fst: TypeId, snd: TypeId },
    Sum { left: TypeId, right: TypeId },
    Array { index: TypeId, elem: TypeId },
    Func { dom: TypeId, cod: TypeId },
}

#[derive(Clone, Copy, Debug)]
enum Expr {
    Undefined,
    Use { module: ModId, id: ValId },
}

#[derive(Clone, Copy, Debug)]
struct Val {
    ty: TypeId,
    expr: Expr,
}

#[derive(Debug)]
pub struct Module {
    imports: Vec<String>,
    types: IndexSet<Type>,
    exports: HashMap<String, ValId>,
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
    Duplicate { name: TokenId },
    Untyped { name: TokenId },
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
            index: self
                .module
                .vals
                .len()
                .try_into()
                .expect("tokens should outnumber values"),
        };
        self.module.vals.push(val);
        id
    }

    fn parse_ty(
        &mut self,
        names: &HashMap<&'a str, TypeId>,
        id: parse::TypeId,
    ) -> Result<TypeId, TypeError> {
        match self.tree.ty(id) {
            parse::Type::Unit => todo!(),
            parse::Type::Name { name } => match self.token(name) {
                "Int" => self.ty(Type::Int),
                "Float" => self.ty(Type::Float),
                s => names.get(s).ok_or(TypeError::Undefined { name }).copied(),
            },
            parse::Type::Prod { fst, snd } => {
                let fst = self.parse_ty(names, fst)?;
                let snd = self.parse_ty(names, snd)?;
                self.ty(Type::Prod { fst, snd })
            }
            parse::Type::Sum { left, right } => {
                let left = self.parse_ty(names, left)?;
                let right = self.parse_ty(names, right)?;
                self.ty(Type::Sum { left, right })
            }
            parse::Type::Array { index, elem } => {
                let index = match index {
                    Some(i) => self.parse_ty(names, i),
                    None => self.ty(Type::Int),
                }?;
                let elem = self.parse_ty(names, elem)?;
                self.ty(Type::Array { index, elem })
            }
            parse::Type::Func { dom, cod } => {
                let dom = self.parse_ty(names, dom)?;
                let cod = self.parse_ty(names, cod)?;
                self.ty(Type::Func { dom, cod })
            }
        }
    }

    fn bind_ty(
        &mut self,
        names: &HashMap<&'a str, TypeId>,
        bind: parse::Bind,
    ) -> Result<TypeId, TypeError> {
        match bind {
            parse::Bind::Unit => self.ty(Type::Unit),
            parse::Bind::Name { name } => Err(TypeError::Untyped { name }),
            parse::Bind::Pair { fst, snd } => todo!(),
            parse::Bind::Record { name, field, rest } => todo!(),
            parse::Bind::End => todo!(),
        }
    }

    fn param_ty(
        &mut self,
        names: &HashMap<&'a str, TypeId>,
        id: parse::ParamId,
    ) -> Result<TypeId, TypeError> {
        let parse::Param { bind, ty } = self.tree.param(id);
        match ty {
            Some(t) => self.parse_ty(names, t),
            None => self.bind_ty(names, bind),
        }
    }

    fn toplevel(&mut self, name: TokenId, val: ValId) -> Result<(), TypeError> {
        let stack = self.names.entry(self.token(name)).or_default();
        if stack.is_empty() {
            stack.push(val);
            Ok(())
        } else {
            Err(TypeError::Duplicate { name })
        }
    }

    fn translate(
        &mut self,
        i: ModId,
        ids: &mut HashMap<TypeId, TypeId>,
        t0: TypeId,
    ) -> Result<TypeId, TypeError> {
        if let Some(&t) = ids.get(&t0) {
            return Ok(t);
        }
        let t = match self.imports[usize::from(i)].types[usize::from(t0)] {
            Type::Var { module, def } => {
                assert!(module.is_none(), "type variable from transitive import");
                self.ty(Type::Var {
                    module: Some(i),
                    def,
                })?
            }
            Type::Unit => self.ty(Type::Unit)?,
            Type::Int => self.ty(Type::Int)?,
            Type::Float => self.ty(Type::Float)?,
            Type::Prod { fst, snd } => {
                let fst = self.translate(i, ids, fst)?;
                let snd = self.translate(i, ids, snd)?;
                self.ty(Type::Prod { fst, snd })?
            }
            Type::Sum { left, right } => {
                let left = self.translate(i, ids, left)?;
                let right = self.translate(i, ids, right)?;
                self.ty(Type::Sum { left, right })?
            }
            Type::Array { index, elem } => {
                let index = self.translate(i, ids, index)?;
                let elem = self.translate(i, ids, elem)?;
                self.ty(Type::Array { index, elem })?
            }
            Type::Func { dom, cod } => {
                let dom = self.translate(i, ids, dom)?;
                let cod = self.translate(i, ids, cod)?;
                self.ty(Type::Func { dom, cod })?
            }
        };
        ids.insert(t0, t);
        Ok(t)
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
            let mut translated = HashMap::new();
            for &token in imports[usize::from(mod_id)].names.iter() {
                let id = module
                    .export(self.token(token))
                    .ok_or(TypeError::Undefined { name: token })?;
                let ty = self.translate(mod_id, &mut translated, module.val(id).ty)?;
                let val = self.val(Val {
                    ty,
                    expr: Expr::Use { module: mod_id, id },
                });
                self.toplevel(token, val)?;
            }
        }
        Ok(())
    }

    fn module(mut self) -> Result<Module, TypeError> {
        self.imports()?;
        for parse::Def {
            name,
            types,
            params,
            ty,
            body: _,
        } in self.tree.defs()
        {
            let generics = types
                .iter()
                .map(|&def| Ok((self.token(def), self.ty(Type::Var { module: None, def })?)))
                .collect::<Result<HashMap<&'a str, TypeId>, TypeError>>()?;
            let mut t = self.parse_ty(&generics, ty.ok_or(TypeError::Untyped { name: *name })?)?;
            for &param in params.iter().rev() {
                let dom = self.param_ty(&generics, param)?;
                t = self.ty(Type::Func { dom, cod: t })?;
            }
            let expr = Expr::Undefined;
            let val = self.val(Val { ty: t, expr });
            self.toplevel(*name, val)?;
            self.module
                .exports
                .insert(self.token(*name).to_owned(), val);
        }
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
            expr: Expr::Undefined,
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
            expr: Expr::Undefined,
        }],
    }
}
