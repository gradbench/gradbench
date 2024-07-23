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
pub struct VarId {
    pub index: u32,
}

impl From<VarId> for usize {
    fn from(id: VarId) -> Self {
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
    Untyped,
    Var {
        src: Option<ModId>,
        def: TokenId,
    },
    Poly {
        var: TypeId,
        inner: TypeId,
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
pub enum Src {
    Undefined,
    Import { src: ModId, id: parse::DefId },
    Param { id: parse::ParamId },
    Expr { id: parse::ExprId },
    Def { id: parse::DefId },
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Val {
    pub ty: TypeId,
    pub src: Src,
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

    fn make(&mut self, field: &str) -> TypeResult<FieldId> {
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

    fn make(&mut self, ty: Type) -> TypeResult<TypeId> {
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
    fields: Fields,
    types: Types,
    vals: Vec<Val>,
    params: Box<[ValId]>,
    exprs: Box<[ValId]>,
    defs: Box<[ValId]>,
    exports: HashMap<String, parse::DefId>,
}

impl Module {
    pub fn field(&self, id: FieldId) -> &str {
        self.fields.get(id)
    }

    pub fn ty(&self, id: TypeId) -> Type {
        self.types.get(id)
    }

    pub fn val(&self, id: ValId) -> Val {
        self.vals[usize::from(id)]
    }

    pub fn param(&self, id: parse::ParamId) -> ValId {
        self.params[usize::from(id)]
    }

    pub fn expr(&self, id: parse::ExprId) -> ValId {
        self.exprs[usize::from(id)]
    }

    pub fn def(&self, id: parse::DefId) -> ValId {
        self.defs[usize::from(id)]
    }

    pub fn export(&self, name: &str) -> Option<parse::DefId> {
        self.exports.get(name).copied()
    }
}

#[derive(Debug)]
pub enum TypeError {
    TooManyImports,
    TooManyFields,
    TooManyTypes,
    Undefined {
        name: TokenId,
    },
    Duplicate {
        name: TokenId,
    },
    Untyped {
        name: TokenId,
    },
    Param {
        id: parse::ParamId,
        expected: TypeId,
        actual: TypeId,
    },
    Expr {
        id: parse::ExprId,
        expected: TypeId,
        actual: TypeId,
    },
    NotPair {
        param: parse::ParamId,
        ty: TypeId,
    },
    NotFunction {
        expr: parse::ExprId,
        ty: TypeId,
    },
}

type TypeResult<T> = Result<T, TypeError>;

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

    fn field(&mut self, name: &str) -> TypeResult<FieldId> {
        self.module.fields.make(name)
    }

    fn ty(&mut self, ty: Type) -> TypeResult<TypeId> {
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

    fn scope<T, U>(
        &mut self,
        x: T,
        bind: impl FnOnce(&mut Self, &T, &mut Vec<&'a str>) -> TypeResult<()>,
        f: impl FnOnce(&mut Self, T) -> TypeResult<U>,
    ) -> TypeResult<U> {
        let mut names = vec![];
        let res = bind(self, &x, &mut names).and_then(|()| f(self, x));
        for name in names.into_iter().rev() {
            self.names
                .get_mut(name)
                .expect("popped name should be present in the map")
                .pop()
                .expect("popped name should have at least one binding");
        }
        res
    }

    fn parse_ty(
        &mut self,
        types: &IndexMap<&'a str, TypeId>,
        id: parse::TypeId,
    ) -> TypeResult<TypeId> {
        match self.tree.ty(id) {
            parse::Type::Paren { inner } => self.parse_ty(types, inner),
            parse::Type::Unit { open: _, close: _ } => self.ty(Type::Unit),
            parse::Type::Name { name } => match self.token(name) {
                "Int" => self.ty(Type::Int),
                "Float" => self.ty(Type::Float),
                s => types.get(s).ok_or(TypeError::Undefined { name }).copied(),
            },
            parse::Type::Prod { fst, snd } => {
                let fst = self.parse_ty(types, fst)?;
                let snd = self.parse_ty(types, snd)?;
                self.ty(Type::Prod { fst, snd })
            }
            parse::Type::Sum { left, right } => {
                let left = self.parse_ty(types, left)?;
                let right = self.parse_ty(types, right)?;
                self.ty(Type::Sum { left, right })
            }
            parse::Type::Array { index, elem } => {
                let index = match index {
                    Some(i) => self.parse_ty(types, i),
                    None => self.ty(Type::Int),
                }?;
                let elem = self.parse_ty(types, elem)?;
                self.ty(Type::Array { index, elem })
            }
            parse::Type::Func { dom, cod } => {
                let dom = self.parse_ty(types, dom)?;
                let cod = self.parse_ty(types, cod)?;
                self.ty(Type::Func { dom, cod })
            }
        }
    }

    fn bind_ty(
        &mut self,
        types: &IndexMap<&'a str, TypeId>,
        bind: parse::Bind,
    ) -> TypeResult<TypeId> {
        match bind {
            parse::Bind::Paren { inner } => self.param_ty(types, inner),
            parse::Bind::Unit { open: _, close: _ } => self.ty(Type::Unit),
            parse::Bind::Name { name } => Err(TypeError::Untyped { name }),
            parse::Bind::Pair { fst, snd } => {
                let fst = self.param_ty(types, fst)?;
                let snd = self.param_ty(types, snd)?;
                self.ty(Type::Prod { fst, snd })
            }
            parse::Bind::Record { name, field, rest } => {
                let mut fields = BTreeMap::new();
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    let ty = self.param_ty(types, v)?;
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
        types: &IndexMap<&'a str, TypeId>,
        id: parse::ParamId,
    ) -> TypeResult<TypeId> {
        let parse::Param { bind, ty } = self.tree.param(id);
        match ty {
            Some(t) => self.parse_ty(types, t),
            None => self.bind_ty(types, bind),
        }
    }

    fn param(
        &mut self,
        types: &IndexMap<&'a str, TypeId>,
        names: &mut Vec<&'a str>,
        id: parse::ParamId,
        ty: TypeId,
    ) -> TypeResult<()> {
        let src = Src::Param { id };
        let val = self.val(Val { src, ty });
        self.module.params[usize::from(id)] = val;
        match self.tree.param(id).bind {
            parse::Bind::Paren { inner } => self.param(types, names, inner, ty),
            parse::Bind::Unit { open: _, close: _ } => {
                let actual = self.ty(Type::Unit)?;
                if actual == ty {
                    Ok(())
                } else {
                    Err(TypeError::Param {
                        id,
                        expected: ty,
                        actual,
                    })
                }
            }
            parse::Bind::Name { name } => {
                let s = self.token(name);
                self.names.entry(s).or_default().push(val);
                names.push(s);
                Ok(())
            }
            parse::Bind::Pair { fst, snd } => match self.module.ty(ty) {
                Type::Prod { fst: dom, snd: cod } => {
                    self.param(types, names, fst, dom)?;
                    self.param(types, names, snd, cod)?;
                    Ok(())
                }
                _ => Err(TypeError::NotPair { param: id, ty }),
            },
            parse::Bind::Record { name, field, rest } => todo!(),
            parse::Bind::End { open: _, close: _ } => {
                let actual = self.ty(Type::End)?;
                if actual == ty {
                    Ok(())
                } else {
                    Err(TypeError::Param {
                        id,
                        expected: ty,
                        actual,
                    })
                }
            }
        }
    }

    fn expr(
        &mut self,
        types: &mut IndexMap<&'a str, TypeId>,
        id: parse::ExprId,
    ) -> TypeResult<ValId> {
        let ty = match self.tree.expr(id) {
            parse::Expr::Paren { inner } => {
                let val = self.expr(types, inner)?;
                self.module.val(val).ty
            }
            parse::Expr::Name { name } => {
                let val = self
                    .names
                    .get(self.token(name))
                    .and_then(|stack| stack.last().copied())
                    .ok_or(TypeError::Undefined { name })?;
                self.module.val(val).ty
            }
            parse::Expr::Undefined { token: _ } => self.ty(Type::Untyped)?,
            parse::Expr::Unit { open: _, close: _ } => todo!(),
            parse::Expr::Number { val } => todo!(),
            parse::Expr::Pair { fst, snd } => todo!(),
            parse::Expr::Record { name, field, rest } => todo!(),
            parse::Expr::End { open: _, close: _ } => todo!(),
            parse::Expr::Elem { array, index } => todo!(),
            parse::Expr::Apply { func, arg } => {
                let f = self.expr(types, func)?;
                let fty = self.module.val(f).ty;
                let (dom, cod) = match self.module.ty(fty) {
                    Type::Func { dom, cod } => Ok((dom, cod)),
                    _ => Err(TypeError::NotFunction { expr: id, ty: fty }),
                }?;
                self.expect(types, arg, dom)?;
                cod
            }
            parse::Expr::Map { func, arg } => todo!(),
            parse::Expr::Let { param, val, body } => {
                let bound = self.expr(types, val)?;
                let xty = self.module.val(bound).ty;
                let rest = self.scope(
                    types,
                    |this, types, names| this.param(types, names, param, xty),
                    |this, types| this.expr(types, body),
                )?;
                self.module.val(rest).ty
            }
            parse::Expr::Index { name, val, body } => todo!(),
            parse::Expr::Unary { op, arg } => todo!(),
            parse::Expr::Binary { lhs, map, op, rhs } => {
                if map {
                    todo!()
                }
                let float = self.ty(Type::Float)?;
                let l = self.expect(types, lhs, float)?;
                let r = self.expect(types, rhs, float)?;
                float
            }
            parse::Expr::Lambda { param, ty, body } => todo!(),
        };
        let src = Src::Expr { id };
        let val = self.val(Val { src, ty });
        self.module.exprs[usize::from(id)] = val;
        Ok(val)
    }

    fn expect(
        &mut self,
        types: &mut IndexMap<&'a str, TypeId>,
        id: parse::ExprId,
        ty: TypeId,
    ) -> TypeResult<ValId> {
        let val = self.expr(types, id)?;
        let t = self.module.val(val).ty;
        if t == ty {
            Ok(val)
        } else {
            Err(TypeError::Expr {
                id,
                expected: ty,
                actual: t,
            })
        }
    }

    fn toplevel(&mut self, name: TokenId, val: ValId) -> TypeResult<()> {
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
    ) -> TypeResult<TypeId> {
        if let Some(&t) = ids.get(&t0) {
            return Ok(t);
        }
        let import = self.imports[usize::from(i)];
        let t = match import.ty(t0) {
            Type::Untyped => self.ty(Type::Untyped)?,
            Type::Var { src, def } => {
                assert!(src.is_none(), "type variable from transitive import");
                self.ty(Type::Var { src: Some(i), def })?
            }
            Type::Poly { var, inner } => {
                let var = self.translate(i, ids, var)?;
                let inner = self.translate(i, ids, inner)?;
                self.ty(Type::Poly { var, inner })?
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

    fn imports(&mut self) -> TypeResult<()> {
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
                let val = module.def(id);
                let ty = self.translate(src, &mut translated, module.val(val).ty)?;
                let val = self.val(Val {
                    ty,
                    src: Src::Import { src, id },
                });
                self.toplevel(token, val)?;
            }
        }
        Ok(())
    }

    fn module(&mut self) -> TypeResult<()> {
        self.imports()?;
        let defs = self
            .tree
            .defs()
            .iter()
            .enumerate()
            .map(|(i, def)| {
                let parse::Def {
                    name,
                    types,
                    params,
                    ty,
                    body: _,
                } = def;
                let names = types
                    .iter()
                    .map(|&def| Ok((self.token(def), self.ty(Type::Var { src: None, def })?)))
                    .collect::<TypeResult<IndexMap<&'a str, TypeId>>>()?;
                let doms = params
                    .iter()
                    .map(|&param| self.param_ty(&names, param))
                    .collect::<TypeResult<Vec<TypeId>>>()?;
                let cod = self.parse_ty(&names, ty.ok_or(TypeError::Untyped { name: *name })?)?;
                let t = names.values().try_rfold(
                    doms.iter()
                        .try_rfold(cod, |cod, &dom| self.ty(Type::Func { dom, cod }))?,
                    |inner, &var| self.ty(Type::Poly { var, inner }),
                )?;
                let id = parse::DefId {
                    index: i.try_into().unwrap(),
                };
                let src = Src::Def { id };
                let val = self.val(Val { ty: t, src });
                self.toplevel(*name, val)?;
                self.module.defs[usize::from(id)] = val;
                self.module.exports.insert(self.token(*name).to_owned(), id);
                Ok((names, doms, cod))
            })
            .collect::<TypeResult<Vec<_>>>()?;
        for (
            parse::Def {
                name: _,
                types: _,
                params,
                ty: _,
                body,
            },
            (types, doms, cod),
        ) in self.tree.defs().iter().zip(defs)
        {
            self.scope(
                types,
                |this, types, names| {
                    for (&param, t) in params.iter().zip(doms) {
                        this.param(types, names, param, t)?;
                    }
                    Ok(())
                },
                |this, mut types| {
                    let ty = if let parse::Expr::Undefined { token: _ } = this.tree.expr(*body) {
                        this.ty(Type::Untyped)? // escape hatch if the entire body is undefined
                    } else {
                        cod
                    };
                    this.expect(&mut types, *body, ty)
                },
            )?;
        }
        Ok(())
    }
}

pub fn typecheck(
    source: &str,
    tokens: &Tokens,
    tree: &parse::Module,
    imports: Vec<&Module>,
) -> (Module, Result<(), TypeError>) {
    let mut types = Types::new();
    let ty = types.make(Type::Untyped).unwrap();
    let src = Src::Undefined;
    let vals = vec![Val { ty, src }];
    let undefined = ValId { index: 0 };
    let mut typer = Typer {
        imports,
        source,
        tokens,
        tree,
        module: Module {
            fields: Fields::new(),
            types,
            vals,
            params: vec![undefined; tree.params().len()].into_boxed_slice(),
            exprs: vec![undefined; tree.exprs().len()].into_boxed_slice(),
            defs: vec![undefined; tree.defs().len()].into_boxed_slice(),
            exports: HashMap::new(),
        },
        names: HashMap::new(),
    };
    let res = typer.module();
    (typer.module, res)
}
