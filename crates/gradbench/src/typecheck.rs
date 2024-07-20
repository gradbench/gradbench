use std::collections::{BTreeMap, HashMap};

use indexmap::{map::RawEntryApiV1, IndexMap, IndexSet};
use serde::{ser::SerializeSeq, Serialize, Serializer};

use crate::{
    lex::{TokenId, Tokens},
    parse,
    util::u32_to_usize,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ModId {
    pub index: u16,
}

impl From<ModId> for usize {
    fn from(id: ModId) -> Self {
        id.index.into()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct FieldId {
    pub index: u32,
}

impl From<FieldId> for usize {
    fn from(id: FieldId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct TypeId {
    pub index: u32,
}

impl From<TypeId> for usize {
    fn from(id: TypeId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ValId {
    pub index: u32,
}

impl From<ValId> for usize {
    fn from(id: ValId) -> Self {
        u32_to_usize(id.index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Type {
    Var {
        src: Option<ModId>,
        def: TokenId,
    },
    Unit,
    Int,
    Float,
    Prod {
        fst: TypeId,
        snd: TypeId,
    },
    Sum {
        left: TypeId,
        right: TypeId,
    },
    Array {
        index: TypeId,
        elem: TypeId,
    },
    Record {
        name: FieldId,
        field: TypeId,
        rest: TypeId,
    },
    End,
    Func {
        dom: TypeId,
        cod: TypeId,
    },
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "kind")]
pub enum Expr {
    Undefined,
    Use { src: ModId, id: ValId },
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Val {
    pub ty: TypeId,
    pub expr: Expr,
}

#[derive(Debug)]
struct Fields {
    fields: IndexMap<String, ()>,
}

impl Fields {
    fn new() -> Self {
        Self {
            fields: IndexMap::new(),
        }
    }

    fn get(&self, id: FieldId) -> &str {
        let (s, _) = self.fields.get_index(usize::from(id)).unwrap();
        s
    }

    fn make(&mut self, field: &str) -> Result<FieldId, TypeError> {
        let entry = self.fields.raw_entry_mut_v1().from_key(field);
        let id = FieldId {
            // maybe more fields than tokens, because of imports
            index: entry
                .index()
                .try_into()
                .map_err(|_| TypeError::TooManyFields)?,
        };
        entry.or_insert_with(|| (field.to_owned(), ()));
        Ok(id)
    }
}

impl Serialize for Fields {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.fields.len()))?;
        for s in self.fields.keys() {
            seq.serialize_element(s)?;
        }
        seq.end()
    }
}

#[derive(Debug)]
struct Types {
    types: IndexSet<Type>,
}

impl Types {
    fn new() -> Self {
        Self {
            types: IndexSet::new(),
        }
    }

    fn get(&self, id: TypeId) -> Type {
        self.types[usize::from(id)]
    }

    fn make(&mut self, ty: Type) -> Result<TypeId, TypeError> {
        let (i, _) = self.types.insert_full(ty);
        let id = TypeId {
            // maybe more types than tokens, because of imports
            index: i.try_into().map_err(|_| TypeError::TooManyTypes)?,
        };
        Ok(id)
    }
}

impl Serialize for Types {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.types.len()))?;
        for ty in &self.types {
            seq.serialize_element(ty)?;
        }
        seq.end()
    }
}

#[derive(Debug, Serialize)]
pub struct Module {
    imports: Vec<String>,
    fields: Fields,
    types: Types,
    exports: HashMap<String, ValId>,
    vals: Vec<Val>,
}

impl Module {
    pub fn field(&self, id: FieldId) -> &str {
        self.fields.get(id)
    }

    pub fn ty(&self, id: TypeId) -> Type {
        self.types.get(id)
    }

    pub fn export(&self, name: &str) -> Option<ValId> {
        self.exports.get(name).copied()
    }

    pub fn val(&self, id: ValId) -> Val {
        self.vals[usize::from(id)]
    }
}

#[derive(Debug)]
pub enum TypeError {
    TooManyImports,
    TooManyFields,
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

    fn field(&mut self, name: &str) -> Result<FieldId, TypeError> {
        self.module.fields.make(name)
    }

    fn ty(&mut self, ty: Type) -> Result<TypeId, TypeError> {
        self.module.types.make(ty)
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
            parse::Type::Paren { inner } => self.parse_ty(names, inner),
            parse::Type::Unit { open: _, close: _ } => self.ty(Type::Unit),
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
            parse::Bind::Paren { inner } => self.param_ty(names, inner),
            parse::Bind::Unit { open: _, close: _ } => self.ty(Type::Unit),
            parse::Bind::Name { name } => Err(TypeError::Untyped { name }),
            parse::Bind::Pair { fst, snd } => {
                let fst = self.param_ty(names, fst)?;
                let snd = self.param_ty(names, snd)?;
                self.ty(Type::Prod { fst, snd })
            }
            parse::Bind::Record { name, field, rest } => {
                let mut fields = BTreeMap::new();
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    let ty = self.param_ty(names, v)?;
                    if fields.insert(self.token(n), ty).is_some() {
                        return Err(TypeError::Duplicate { name: n });
                    }
                    match self.tree.param(r).bind {
                        parse::Bind::Record { name, field, rest } => {
                            (n, v, r) = (name, field, rest);
                        }
                        parse::Bind::End { open: _, close: _ } => break,
                        _ => panic!("invalid record"),
                    }
                }
                fields
                    .into_iter()
                    .try_rfold(self.ty(Type::End)?, |rest, (s, field)| {
                        let name = self.field(s)?;
                        self.ty(Type::Record { name, field, rest })
                    })
            }
            parse::Bind::End { open: _, close: _ } => self.ty(Type::End),
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
        let import = self.imports[usize::from(i)];
        let t = match import.ty(t0) {
            Type::Var { src, def } => {
                assert!(src.is_none(), "type variable from transitive import");
                self.ty(Type::Var { src: Some(i), def })?
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
            Type::Record { name, field, rest } => {
                let name = self.field(import.field(name))?;
                let field = self.translate(i, ids, field)?;
                let rest = self.translate(i, ids, rest)?;
                self.ty(Type::Record { name, field, rest })?
            }
            Type::End => self.ty(Type::End)?,
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
            let src = ModId { index };
            let module = self.imports[usize::from(src)];
            let mut translated = HashMap::new();
            for &token in imports[usize::from(src)].names.iter() {
                let id = module
                    .export(self.token(token))
                    .ok_or(TypeError::Undefined { name: token })?;
                let ty = self.translate(src, &mut translated, module.val(id).ty)?;
                let val = self.val(Val {
                    ty,
                    expr: Expr::Use { src, id },
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
                .map(|&def| Ok((self.token(def), self.ty(Type::Var { src: None, def })?)))
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
            fields: Fields::new(),
            types: Types::new(),
            exports: HashMap::new(),
            vals: vec![],
        },
        names: HashMap::new(),
    }
    .module()
}

pub fn array() -> Module {
    let mut types = Types::new();
    let unit = types.make(Type::Unit).unwrap();
    let func = types
        .make(Type::Func {
            dom: unit,
            cod: unit,
        })
        .unwrap();
    Module {
        imports: vec![],
        fields: Fields::new(),
        types,
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
        vals: vec![Val {
            ty: func,
            expr: Expr::Undefined,
        }],
    }
}

pub fn math() -> Module {
    let mut types = Types::new();
    let unit = types.make(Type::Unit).unwrap();
    let func = types
        .make(Type::Func {
            dom: unit,
            cod: unit,
        })
        .unwrap();
    Module {
        imports: vec![],
        fields: Fields::new(),
        types,
        exports: HashMap::from(
            ["exp", "int", "lgamma", "log", "pi", "sqr", "sqrt"]
                .map(|s| (s.to_owned(), ValId { index: 0 })),
        ),
        vals: vec![Val {
            ty: func,
            expr: Expr::Undefined,
        }],
    }
}
