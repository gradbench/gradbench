use std::collections::{BTreeMap, HashMap};

use indexmap::{map::RawEntryApiV1, IndexMap, IndexSet};
use serde::{ser::SerializeSeq, Serialize, Serializer};

use crate::{
    lex::{TokenId, Tokens},
    parse,
    util::{u32_to_usize, Id},
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ImportId {
    pub index: u16,
}

impl Id for ImportId {
    fn from_usize(n: usize) -> Option<Self> {
        match n.try_into() {
            Ok(index) => Some(Self { index }),
            Err(_) => None,
        }
    }

    fn to_usize(self) -> usize {
        self.index.into()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct FieldId {
    pub index: u32,
}

impl Id for FieldId {
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
pub struct TypeId {
    pub index: u32,
}

impl Id for TypeId {
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
pub struct UnknownId {
    pub index: u32,
}

impl Id for UnknownId {
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
pub struct VarId {
    pub index: u32,
}

impl Id for VarId {
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
pub struct ValId {
    pub index: u32,
}

impl Id for ValId {
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Type {
    Unknown {
        id: UnknownId,
    },
    Scalar {
        id: UnknownId,
    },
    Vector {
        id: UnknownId,
        scalar: TypeId,
    },
    Fragment,
    Var {
        src: Option<ImportId>,
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Src {
    Import { src: ImportId, id: parse::DefId },
    Param { id: parse::ParamId },
    Expr { id: parse::ExprId },
    Def { id: parse::DefId },
    Inst { val: ValId, ty: TypeId },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
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
        let (s, _) = self.fields.get_index(id.to_usize()).unwrap();
        s
    }

    fn make(&mut self, field: &str) -> TypeResult<FieldId> {
        let entry = self.fields.raw_entry_mut_v1().from_key(field);
        // maybe more fields than tokens, because of imports
        let id = FieldId::from_usize(entry.index()).ok_or(TypeError::TooManyFields)?;
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
enum BasicError {
    TooManyTypes,
    FailedToUnify,
}

#[derive(Debug)]
struct Types {
    unknowns: usize,
    types: IndexMap<Type, TypeId>,
}

impl Types {
    fn new() -> Self {
        Self {
            unknowns: 0,
            types: IndexMap::new(),
        }
    }

    fn get(&self, id: TypeId) -> Type {
        let (&t, _) = self.types.get_index(id.to_usize()).unwrap();
        t
    }

    fn make(&mut self, ty: Type) -> Result<TypeId, BasicError> {
        let entry = self.types.entry(ty);
        // maybe more types than tokens, because of imports
        let parent = TypeId::from_usize(entry.index()).ok_or(BasicError::TooManyTypes)?;
        entry.or_insert(parent);
        Ok(parent)
    }

    fn unknown(&mut self, f: impl FnOnce(UnknownId) -> Type) -> Result<TypeId, BasicError> {
        let t = self.make(f(
            UnknownId::from_usize(self.unknowns).expect("types should outnumber unknowns")
        ))?;
        self.unknowns += 1;
        Ok(t)
    }

    fn parent(&self, element: TypeId) -> TypeId {
        self.types[element.to_usize()]
    }

    fn set_parent(&mut self, element: TypeId, parent: TypeId) {
        self.types[element.to_usize()] = parent;
    }

    fn root(&mut self, mut element: TypeId) -> TypeId {
        let mut parent = self.parent(element);
        while parent != element {
            let grandparent = self.parent(parent);
            self.set_parent(element, grandparent);
            element = parent;
            parent = grandparent;
        }
        element
    }

    fn unify(&mut self, t1: TypeId, t2: TypeId) -> Result<TypeId, BasicError> {
        let (t1, t2) = (self.root(t1), self.root(t2));
        if t1 == t2 {
            return Ok(t1);
        }
        let t = match (self.get(t1), self.get(t2)) {
            (_, Type::Unknown { id: _ }) => t1,
            (Type::Unknown { id: _ }, _) => t2,
            (Type::Scalar { id: _ }, Type::Scalar { id: _ }) => t1,
            (Type::Int | Type::Float, Type::Scalar { id: _ }) => t1,
            (Type::Scalar { id: _ }, Type::Int | Type::Float) => t2,
            (Type::Vector { id: _, scalar: _ }, Type::Vector { id: _, scalar: _ }) => t1,
            (Type::Int | Type::Float, Type::Vector { id: _, scalar }) => self.unify(t1, scalar)?,
            (Type::Vector { id: _, scalar }, Type::Int | Type::Float) => self.unify(scalar, t2)?,
            (Type::Array { index, elem }, Type::Vector { id: _, scalar })
            | (Type::Vector { id: _, scalar }, Type::Array { index, elem }) => {
                let elem = self.unify(elem, scalar)?;
                self.make(Type::Array { index, elem })?
            }
            (
                Type::Prod {
                    fst: fst1,
                    snd: snd1,
                },
                Type::Prod {
                    fst: fst2,
                    snd: snd2,
                },
            ) => {
                let fst = self.unify(fst1, fst2)?;
                let snd = self.unify(snd1, snd2)?;
                self.make(Type::Prod { fst, snd })?
            }
            (
                Type::Sum {
                    left: left1,
                    right: right1,
                },
                Type::Sum {
                    left: left2,
                    right: right2,
                },
            ) => {
                let left = self.unify(left1, left2)?;
                let right = self.unify(right1, right2)?;
                self.make(Type::Sum { left, right })?
            }
            (
                Type::Array {
                    index: index1,
                    elem: elem1,
                },
                Type::Array {
                    index: index2,
                    elem: elem2,
                },
            ) => {
                let index = self.unify(index1, index2)?;
                let elem = self.unify(elem1, elem2)?;
                self.make(Type::Array { index, elem })?
            }
            (
                Type::Record {
                    name: name1,
                    field: field1,
                    rest: rest1,
                },
                Type::Record {
                    name: name2,
                    field: field2,
                    rest: rest2,
                },
            ) => {
                if name1 != name2 {
                    return Err(BasicError::FailedToUnify);
                }
                let name = name1;
                let field = self.unify(field1, field2)?;
                let rest = self.unify(rest1, rest2)?;
                self.make(Type::Record { name, field, rest })?
            }
            (
                Type::Func {
                    dom: dom1,
                    cod: cod1,
                },
                Type::Func {
                    dom: dom2,
                    cod: cod2,
                },
            ) => {
                let dom = self.unify(dom1, dom2)?;
                let cod = self.unify(cod1, cod2)?;
                self.make(Type::Func { dom, cod })?
            }
            _ => return Err(BasicError::FailedToUnify),
        };
        self.set_parent(t1, t);
        self.set_parent(t2, t);
        Ok(t)
    }
}

impl Serialize for Types {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.types.len()))?;
        for ty in self.types.keys() {
            seq.serialize_element(ty)?;
        }
        seq.end()
    }
}

#[derive(Debug)]
struct Canonizer {
    errors: Vec<TypeError>,
    types: HashMap<TypeId, TypeId>,
    old_types: Types,
    new_types: Types,
    vals: HashMap<ValId, ValId>,
    old_vals: Vec<Val>,
    new_vals: IndexSet<Val>,
}

impl Canonizer {
    fn make(&mut self, ty: Type) -> TypeId {
        self.new_types
            .make(ty)
            .expect("old types should outnumber new types")
    }

    fn unknown(&mut self, f: impl FnOnce(UnknownId) -> Type) -> (bool, TypeId) {
        let t = self
            .new_types
            .unknown(f)
            .expect("old types should outnumber new types");
        (true, t)
    }

    fn ty(&mut self, t0: TypeId) -> (bool, TypeId) {
        let t0 = self.old_types.root(t0);
        if let Some(&t) = self.types.get(&t0) {
            return (false, t);
        }
        let (ambiguous, t) = match self.old_types.get(t0) {
            Type::Unknown { id: _ } => self.unknown(|id| Type::Unknown { id }),
            Type::Scalar { id: _ } => self.unknown(|id| Type::Scalar { id }),
            Type::Vector { id: _, scalar } => {
                let (_, scalar) = self.ty(scalar);
                self.unknown(|id| Type::Vector { id, scalar })
            }
            Type::Fragment => (false, self.make(Type::Fragment)),
            Type::Var { src, def } => (false, self.make(Type::Var { src, def })),
            Type::Poly { var, inner } => {
                let (a1, var) = self.ty(var);
                let (a2, inner) = self.ty(inner);
                (a1 || a2, self.make(Type::Poly { var, inner }))
            }
            Type::Unit => (false, self.make(Type::Unit)),
            Type::Int => (false, self.make(Type::Int)),
            Type::Float => (false, self.make(Type::Float)),
            Type::Prod { fst, snd } => {
                let (a1, fst) = self.ty(fst);
                let (a2, snd) = self.ty(snd);
                (a1 || a2, self.make(Type::Prod { fst, snd }))
            }
            Type::Sum { left, right } => {
                let (a1, left) = self.ty(left);
                let (a2, right) = self.ty(right);
                (a1 || a2, self.make(Type::Sum { left, right }))
            }
            Type::Array { index, elem } => {
                let (a1, index) = self.ty(index);
                let (a2, elem) = self.ty(elem);
                (a1 || a2, self.make(Type::Array { index, elem }))
            }
            Type::Record { name, field, rest } => {
                let (a1, field) = self.ty(field);
                let (a2, rest) = self.ty(rest);
                (a1 || a2, self.make(Type::Record { name, field, rest }))
            }
            Type::End => (false, self.make(Type::End)),
            Type::Func { dom, cod } => {
                let (a1, dom) = self.ty(dom);
                let (a2, cod) = self.ty(cod);
                (a1 || a2, self.make(Type::Func { dom, cod }))
            }
        };
        self.types.insert(t0, t);
        (ambiguous, t)
    }

    fn val(&mut self, v0: ValId) -> (bool, bool, ValId) {
        let mut ambig_type_args = false;
        if let Some(&v) = self.vals.get(&v0) {
            return (ambig_type_args, false, v);
        }
        let Val { ty, mut src } = self.old_vals[v0.to_usize()];
        if let Src::Inst { val, ty } = src {
            let (ata, _, val) = self.val(val);
            ambig_type_args |= ata;
            let (amb, ty) = self.ty(ty);
            ambig_type_args |= amb;
            src = Src::Inst { val, ty };
        }
        let (ambiguous, ty) = self.ty(ty);
        let (i, _) = self.new_vals.insert_full(Val { ty, src });
        let v = ValId::from_usize(i).expect("old values should outnumber new values");
        self.vals.insert(v0, v);
        (ambig_type_args, ambiguous, v)
    }
}

#[derive(Debug, Serialize)]
pub struct Module {
    fields: Fields,
    types: Types,
    vals: Vec<Val>,
    params: Vec<ValId>,
    exprs: Vec<ValId>,
    defs: Vec<ValId>,
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
        self.vals[id.to_usize()]
    }

    pub fn param(&self, id: parse::ParamId) -> ValId {
        self.params[id.to_usize()]
    }

    pub fn expr(&self, id: parse::ExprId) -> ValId {
        self.exprs[id.to_usize()]
    }

    pub fn def(&self, id: parse::DefId) -> ValId {
        self.defs[id.to_usize()]
    }

    pub fn export(&self, name: &str) -> Option<parse::DefId> {
        self.exports.get(name).copied()
    }

    fn set_expr(&mut self, id: parse::ExprId, val: ValId) {
        self.exprs[id.to_usize()] = val;
    }

    fn gc(mut self) -> (Self, Vec<TypeError>) {
        let mut canonizer = Canonizer {
            errors: vec![],
            types: HashMap::new(),
            old_types: self.types,
            new_types: Types::new(),
            vals: HashMap::new(),
            old_vals: self.vals,
            new_vals: IndexSet::new(),
        };
        self.defs = self
            .defs
            .into_iter()
            .map(|v0| {
                let (ambig_type_args, ambiguous, v) = canonizer.val(v0);
                assert!(!ambig_type_args, "top-level cannot be a type instantiation");
                assert!(!ambiguous, "top-level types should already be resolved");
                v
            })
            .collect();
        self.params = self
            .params
            .into_iter()
            .enumerate()
            .map(|(i, v0)| {
                let (ambig_type_args, ambiguous, v) = canonizer.val(v0);
                assert!(!ambig_type_args, "parameter cannot be a type instantiation");
                if ambiguous {
                    canonizer.errors.push(TypeError::AmbigParam {
                        id: parse::ParamId::from_usize(i).unwrap(),
                    });
                }
                v
            })
            .collect();
        self.exprs = self
            .exprs
            .into_iter()
            .enumerate()
            .map(|(i, v0)| {
                let (ambig_type_args, ambiguous, v) = canonizer.val(v0);
                let id = parse::ExprId::from_usize(i).unwrap();
                if ambig_type_args {
                    canonizer.errors.push(TypeError::AmbigTypeArgs { id });
                } else if ambiguous {
                    // these can still happen in the case of an ambiguous generic function
                    // application with some (but not all) manually specified type arguments, but in
                    // that case the overall function application should be flagged as ambiguous by
                    // the above, so we don't do anything else here
                }
                v
            })
            .collect();
        self.types = canonizer.new_types;
        self.vals = canonizer.new_vals.into_iter().collect();
        (self, canonizer.errors)
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
    Param { id: parse::ParamId },
    Elem { id: parse::ExprId },
    Inst { id: parse::ExprId, n: usize },
    Apply { id: parse::ExprId },
    MapLhs { id: parse::ExprId },
    MapRhs { id: parse::ExprId },
    Let { id: parse::ExprId },
    Index { id: parse::ExprId },
    Neg { id: parse::ExprId },
    ElemLhs { id: parse::ExprId },
    ElemRhs { id: parse::ExprId },
    MulLhs { id: parse::ExprId },
    MulRhs { id: parse::ExprId },
    DivLhs { id: parse::ExprId },
    DivRhs { id: parse::ExprId },
    Lambda { id: parse::ExprId },
    Def { id: parse::DefId },
    AmbigParam { id: parse::ParamId },
    AmbigTypeArgs { id: parse::ExprId },
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
        self.module.types.make(ty).map_err(|e| match e {
            BasicError::TooManyTypes => TypeError::TooManyTypes,
            BasicError::FailedToUnify => unreachable!(),
        })
    }

    fn unknown(&mut self) -> TypeResult<TypeId> {
        self.module
            .types
            .unknown(|id| Type::Unknown { id })
            .map_err(|e| match e {
                BasicError::TooManyTypes => TypeError::TooManyTypes,
                BasicError::FailedToUnify => unreachable!(),
            })
    }

    fn scalar(&mut self) -> TypeResult<TypeId> {
        self.module
            .types
            .unknown(|id| Type::Scalar { id })
            .map_err(|e| match e {
                BasicError::TooManyTypes => TypeError::TooManyTypes,
                BasicError::FailedToUnify => unreachable!(),
            })
    }

    fn vector(&mut self, scalar: TypeId) -> TypeResult<TypeId> {
        self.module
            .types
            .unknown(|id| Type::Vector { id, scalar })
            .map_err(|e| match e {
                BasicError::TooManyTypes => TypeError::TooManyTypes,
                BasicError::FailedToUnify => unreachable!(),
            })
    }

    fn root(&mut self, ty: TypeId) -> TypeId {
        self.module.types.root(ty)
    }

    fn unify(
        &mut self,
        t1: TypeId,
        t2: TypeId,
        err: impl FnOnce() -> TypeError,
    ) -> TypeResult<TypeId> {
        self.module.types.unify(t1, t2).map_err(|e| match e {
            BasicError::TooManyTypes => TypeError::TooManyTypes,
            BasicError::FailedToUnify => err(),
        })
    }

    fn unify_assert(&mut self, t1: TypeId, t2: TypeId) -> TypeResult<TypeId> {
        self.unify(t1, t2, || panic!("expected unification to succeed"))
    }

    fn unknowns<T>(&mut self, a: &[T], mut f: impl FnMut(usize) -> Src) -> TypeResult<Vec<ValId>> {
        (0..a.len())
            .map(|i| {
                let ty = self.unknown()?;
                let src = f(i);
                Ok(self.val(Val { ty, src }))
            })
            .collect()
    }

    fn sub(&mut self, var: TypeId, inner: TypeId, ty: TypeId) -> TypeResult<TypeId> {
        match self.module.ty(inner) {
            Type::Unknown { id: _ }
            | Type::Scalar { id: _ }
            | Type::Unit
            | Type::Int
            | Type::Float
            | Type::End => Ok(inner),
            Type::Vector { id, scalar } => {
                let scalar = self.sub(var, scalar, ty)?;
                self.ty(Type::Vector { id, scalar })
            }
            Type::Fragment => self.ty(Type::Fragment),
            Type::Var { src: _, def: _ } => Ok(if inner == var { ty } else { inner }),
            Type::Poly { var: x, inner: t } => {
                assert_ne!(x, var, "type variables should be unique");
                let t = self.sub(var, t, ty)?;
                self.ty(Type::Poly { var: x, inner: t })
            }
            Type::Prod { fst, snd } => {
                let fst = self.sub(var, fst, ty)?;
                let snd = self.sub(var, snd, ty)?;
                self.ty(Type::Prod { fst, snd })
            }
            Type::Sum { left, right } => {
                let left = self.sub(var, left, ty)?;
                let right = self.sub(var, right, ty)?;
                self.ty(Type::Sum { left, right })
            }
            Type::Array { index, elem } => {
                let index = self.sub(var, index, ty)?;
                let elem = self.sub(var, elem, ty)?;
                self.ty(Type::Array { index, elem })
            }
            Type::Record { name, field, rest } => {
                let field = self.sub(var, field, ty)?;
                let rest = self.sub(var, rest, ty)?;
                self.ty(Type::Record { name, field, rest })
            }
            Type::Func { dom, cod } => {
                let dom = self.sub(var, dom, ty)?;
                let cod = self.sub(var, cod, ty)?;
                self.ty(Type::Func { dom, cod })
            }
        }
    }

    fn val(&mut self, val: Val) -> ValId {
        let id = ValId::from_usize(self.module.vals.len()).expect("tokens should outnumber values");
        self.module.vals.push(val);
        id
    }

    fn scope<A, B, C>(
        &mut self,
        x: A,
        bind: impl FnOnce(&mut Self, &A, &mut Vec<&'a str>) -> TypeResult<B>,
        f: impl FnOnce(&mut Self, A) -> TypeResult<C>,
    ) -> TypeResult<(B, C)> {
        let mut names = vec![];
        let res = bind(self, &x, &mut names).and_then(|a| {
            let b = f(self, x)?;
            Ok((a, b))
        });
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

    fn param(
        &mut self,
        types: &IndexMap<&'a str, TypeId>,
        names: &mut Vec<&'a str>,
        id: parse::ParamId,
    ) -> TypeResult<TypeId> {
        let parse::Param { bind, ty } = self.tree.param(id);
        let val = self.module.param(id);
        let unknown = self.module.val(val).ty;
        let actual = match bind {
            parse::Bind::Paren { inner } => self.param(types, names, inner)?,
            parse::Bind::Unit { open: _, close: _ } => self.ty(Type::Unit)?,
            parse::Bind::Name { name } => {
                let s = self.token(name);
                self.names.entry(s).or_default().push(val);
                names.push(s);
                unknown
            }
            parse::Bind::Pair { fst, snd } => {
                let fst = self.param(types, names, fst)?;
                let snd = self.param(types, names, snd)?;
                self.ty(Type::Prod { fst, snd })?
            }
            parse::Bind::Record { name, field, rest } => {
                let fragment = self.ty(Type::Fragment)?;
                let mut fields = BTreeMap::new();
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    let partial = self.module.val(self.module.param(r)).ty;
                    self.unify_assert(fragment, partial)?;
                    let ty = self.param(types, names, v)?;
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
                    })?
            }
            parse::Bind::End { open: _, close: _ } => self.ty(Type::End)?,
        };
        let expected = match ty {
            Some(ast) => self.parse_ty(types, ast)?,
            None => unknown,
        };
        self.unify(expected, actual, || TypeError::Param { id })
    }

    fn func(
        &mut self,
        types: &mut IndexMap<&'a str, TypeId>,
        type_args: &mut Vec<TypeId>,
        func: parse::ExprId,
    ) -> TypeResult<TypeId> {
        if let parse::Expr::Name { name } = self.tree.expr(func) {
            let mut val = self
                .names
                .get(self.token(name))
                .and_then(|stack| stack.last().copied())
                .ok_or(TypeError::Undefined { name })?;
            let Val { src, mut ty } = self.module.val(val);
            if let Src::Import { src: _, id: _ } | Src::Def { id: _ } = src {
                assert!(self.root(ty) == ty, "top-level type should be resolved");
                while let Type::Poly { var, inner } = self.module.ty(ty) {
                    let t = match type_args.pop() {
                        Some(t) => t,
                        None => self.unknown()?,
                    };
                    ty = self.sub(var, inner, t)?;
                    let src = Src::Inst { val, ty: t };
                    val = self.val(Val { ty, src });
                }
                self.module.set_expr(func, val);
                return Ok(ty);
            }
        }
        self.expr(types, func)
    }

    fn expr(
        &mut self,
        types: &mut IndexMap<&'a str, TypeId>,
        id: parse::ExprId,
    ) -> TypeResult<TypeId> {
        let val = self.module.expr(id);
        let unknown = self.module.val(val).ty;
        assert_eq!(self.root(unknown), unknown, "new expr should have no type");
        match self.tree.expr(id) {
            parse::Expr::Paren { inner } => {
                let ty = self.expr(types, inner)?;
                self.unify_assert(ty, unknown)
            }
            parse::Expr::Name { name } => {
                let val = self
                    .names
                    .get(self.token(name))
                    .and_then(|stack| stack.last().copied())
                    .ok_or(TypeError::Undefined { name })?;
                let Val { src, ty } = self.module.val(val);
                let i = self.module.expr(id).to_usize();
                self.module.vals[i].src = src;
                self.unify_assert(ty, unknown)
            }
            parse::Expr::Undefined { token: _ } => Ok(unknown),
            parse::Expr::Unit { open: _, close: _ } => {
                let unit = self.ty(Type::Unit)?;
                self.unify_assert(unit, unknown)
            }
            parse::Expr::Number { val } => {
                let ty = if self.token(val).contains('.') {
                    self.ty(Type::Float)?
                } else {
                    self.ty(Type::Int)?
                };
                self.unify_assert(ty, unknown)
            }
            parse::Expr::Pair { fst, snd } => {
                let fst = self.expr(types, fst)?;
                let snd = self.expr(types, snd)?;
                let prod = self.ty(Type::Prod { fst, snd })?;
                self.unify_assert(prod, unknown)
            }
            parse::Expr::Record { name, field, rest } => {
                let fragment = self.ty(Type::Fragment)?;
                let mut fields = BTreeMap::new();
                let (mut n, mut v, mut r) = (name, field, rest);
                loop {
                    let partial = self.module.val(self.module.expr(r)).ty;
                    self.unify_assert(fragment, partial)?;
                    let ty = self.expr(types, v)?;
                    if fields.insert(self.token(n), ty).is_some() {
                        return Err(TypeError::Duplicate { name: n });
                    }
                    match self.tree.expr(r) {
                        parse::Expr::Record { name, field, rest } => {
                            (n, v, r) = (name, field, rest);
                        }
                        parse::Expr::End { open: _, close: _ } => break,
                        _ => panic!("invalid record"),
                    }
                }
                let ty =
                    fields
                        .into_iter()
                        .try_rfold(self.ty(Type::End)?, |rest, (s, field)| {
                            let name = self.field(s)?;
                            self.ty(Type::Record { name, field, rest })
                        })?;
                self.unify_assert(ty, unknown)
            }
            parse::Expr::End { open: _, close: _ } => {
                let end = self.ty(Type::End)?;
                self.unify_assert(end, unknown)
            }
            parse::Expr::Elem { array, index } => {
                let index = self.expr(types, index)?;
                let array = self.expr(types, array)?;
                let elem = unknown;
                let expected = self.ty(Type::Array { index, elem })?;
                self.unify(expected, array, || TypeError::Elem { id })?;
                Ok(elem)
            }
            parse::Expr::Inst { val: _, ty: _ } => {
                panic!("polymorphic instantiation should always be inside function application")
            }
            parse::Expr::Apply { mut func, arg } => {
                let fragment = self.ty(Type::Fragment)?;
                let inst = func;
                let mut type_args = vec![];
                while let parse::Expr::Inst { val, ty } = self.tree.expr(func) {
                    let partial = self.module.val(self.module.expr(func)).ty;
                    self.unify_assert(fragment, partial)?;
                    type_args.push(self.parse_ty(types, ty)?);
                    func = val;
                }
                let fty = self.func(types, &mut type_args, func)?;
                if !type_args.is_empty() {
                    return Err(TypeError::Inst {
                        id: inst,
                        n: type_args.len(),
                    });
                }
                let dom = self.expr(types, arg)?;
                let cod = unknown;
                let expected = self.ty(Type::Func { dom, cod })?;
                self.unify(expected, fty, || TypeError::Apply { id })?;
                Ok(cod)
            }
            parse::Expr::Map { func, arg } => {
                if let parse::Expr::Inst { val: _, ty: _ } = self.tree.expr(func) {
                    panic!("dot application can't have explicit type arguments");
                }
                let fty = self.func(types, &mut vec![], func)?;
                let aty = self.expr(types, arg)?;
                let dom = self.unknown()?;
                let cod = self.unknown()?;
                let index = self.unknown()?;
                let f = self.ty(Type::Func { dom, cod })?;
                let a = self.ty(Type::Array { index, elem: dom })?;
                self.unify(f, fty, || TypeError::MapLhs { id })?;
                self.unify(a, aty, || TypeError::MapRhs { id })?;
                let b = self.ty(Type::Array { index, elem: cod })?;
                self.unify_assert(b, unknown)
            }
            parse::Expr::Let { param, val, body } => {
                let actual = self.expr(types, val)?;
                let ((), ty) = self.scope(
                    types,
                    |this, types, names| {
                        let expected = this.param(types, names, param)?;
                        this.unify(expected, actual, || TypeError::Let { id })?;
                        Ok(())
                    },
                    |this, types| this.expr(types, body),
                )?;
                self.unify_assert(ty, unknown)
            }
            parse::Expr::Index { name, val, body } => {
                let expected = self.ty(Type::Int)?;
                let actual = self.expr(types, val)?;
                self.unify(expected, actual, || TypeError::Index { id })?;
                let s = self.token(name);
                let t = self.ty(Type::Var {
                    src: None,
                    def: name,
                })?;
                if types.insert(s, t).is_some() {
                    return Err(TypeError::Duplicate { name });
                }
                let res = self.expr(types, body);
                assert_eq!(types.pop(), Some((s, t)));
                self.unify_assert(res?, unknown)
            }
            parse::Expr::Unary { op, arg } => match op {
                parse::Unop::Neg => {
                    let arg = self.expr(types, arg)?;
                    let scalar = self.scalar()?;
                    let vector = self.vector(scalar)?;
                    self.unify(vector, arg, || TypeError::Neg { id })?;
                    self.unify_assert(vector, unknown)
                }
            },
            parse::Expr::Binary { lhs, op, rhs } => match op {
                parse::Binop::Add
                | parse::Binop::Sub
                | parse::Binop::ElemMul
                | parse::Binop::ElemDiv => {
                    let left = self.expr(types, lhs)?;
                    let right = self.expr(types, rhs)?;
                    let scalar = self.scalar()?;
                    let vector = self.vector(scalar)?;
                    self.unify(vector, left, || TypeError::ElemLhs { id })?;
                    self.unify(vector, right, || TypeError::ElemRhs { id })?;
                    self.unify_assert(vector, unknown)
                }
                parse::Binop::Mul => {
                    let left = self.expr(types, lhs)?;
                    let right = self.expr(types, rhs)?;
                    let scalar = self.scalar()?;
                    let vector = self.vector(scalar)?;
                    self.unify(scalar, left, || TypeError::MulLhs { id })?;
                    self.unify(vector, right, || TypeError::MulRhs { id })?;
                    self.unify_assert(vector, unknown)
                }
                parse::Binop::Div => {
                    let left = self.expr(types, lhs)?;
                    let right = self.expr(types, rhs)?;
                    let scalar = self.scalar()?;
                    let vector = self.vector(scalar)?;
                    self.unify(vector, left, || TypeError::DivLhs { id })?;
                    self.unify(scalar, right, || TypeError::DivRhs { id })?;
                    self.unify_assert(vector, unknown)
                }
            },
            parse::Expr::Lambda { param, ty, body } => {
                let (dom, cod) = self.scope(
                    types,
                    |this, types, names| this.param(types, names, param),
                    |this, types| {
                        let actual = this.expr(types, body)?;
                        let expected = match ty {
                            Some(t) => this.parse_ty(types, t)?,
                            None => actual,
                        };
                        this.unify(expected, actual, || TypeError::Lambda { id })
                    },
                )?;
                let fty = self.ty(Type::Func { dom, cod })?;
                self.unify_assert(fty, unknown)
            }
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
        i: ImportId,
        ids: &mut HashMap<TypeId, TypeId>,
        t0: TypeId,
    ) -> TypeResult<TypeId> {
        if let Some(&t) = ids.get(&t0) {
            return Ok(t);
        }
        let import = self.imports[i.to_usize()];
        let t = match import.ty(t0) {
            Type::Unknown { id: _ }
            | Type::Scalar { id: _ }
            | Type::Vector { id: _, scalar: _ } => {
                panic!("unresolved type from import")
            }
            Type::Fragment => panic!("fragment type from import"),
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
        let imports = self.tree.imports();
        let n = imports.len();
        if n == 0 {
            return Ok(()); // avoid underflow when we decrement below
        }
        for index in 0..=(n - 1).try_into().map_err(|_| TypeError::TooManyImports)? {
            let src = ImportId { index };
            let module = self.imports[src.to_usize()];
            let mut translated = HashMap::new();
            for &token in imports[src.to_usize()].names.iter() {
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
        self.module.params = self.unknowns(self.tree.params(), |i| Src::Param {
            id: parse::ParamId::from_usize(i).unwrap(),
        })?;
        self.module.exprs = self.unknowns(self.tree.exprs(), |i| Src::Expr {
            id: parse::ExprId::from_usize(i).unwrap(),
        })?;
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
                let (doms, ()) = self.scope(
                    &names,
                    |this, names, temps| {
                        params
                            .iter()
                            .map(|&param| this.param(names, temps, param))
                            .collect::<TypeResult<Vec<TypeId>>>()
                    },
                    |_, _| Ok(()),
                )?;
                let cod = self.parse_ty(&names, ty.ok_or(TypeError::Untyped { name: *name })?)?;
                let t = names.values().try_rfold(
                    doms.into_iter()
                        .try_rfold(cod, |cod, dom| self.ty(Type::Func { dom, cod }))?,
                    |inner, &var| self.ty(Type::Poly { var, inner }),
                )?;
                let id = parse::DefId::from_usize(i).unwrap();
                let src = Src::Def { id };
                let val = self.val(Val { ty: t, src });
                self.module.defs.push(val);
                self.toplevel(*name, val)?;
                self.module.exports.insert(self.token(*name).to_owned(), id);
                Ok((id, names, cod))
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
            (id, types, expected),
        ) in self.tree.defs().iter().zip(defs)
        {
            self.scope(
                types,
                |this, types, names| {
                    for &param in params {
                        this.param(types, names, param)?;
                    }
                    Ok(())
                },
                |this, mut types| {
                    let actual = this.expr(&mut types, *body)?;
                    this.unify(expected, actual, || TypeError::Def { id })
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
) -> (Module, Vec<TypeError>) {
    let mut typer = Typer {
        imports,
        source,
        tokens,
        tree,
        module: Module {
            fields: Fields::new(),
            types: Types::new(),
            vals: vec![],
            params: vec![],
            exprs: vec![],
            defs: vec![],
            exports: HashMap::new(),
        },
        names: HashMap::new(),
    };
    let res = typer.module();
    let (module, errors) = typer.module.gc();
    let errs = match res {
        Ok(()) => errors,
        Err(e) => vec![e],
    };
    if module.types.unknowns > 0 {
        assert!(!errs.is_empty(), "ambiguous types should cause errors");
    }
    (module, errs)
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, ops::Range, path::Path};

    use goldenfile::{differs::Differ, Mint};
    use line_index::{LineCol, LineIndex, TextSize};
    use parse::parse;

    use crate::{
        compile::{FullModule, Modules, Printer},
        lex::lex,
        util::{Diagnostic, Emitter},
    };

    use super::*;

    #[derive(Debug)]
    struct IgnoreRelated;

    impl<'a> Diagnostic<(&'a str, Range<usize>)> for IgnoreRelated {
        fn related(&mut self, _: (&'a str, Range<usize>), _: impl ToString) {}

        fn finish(self) {}
    }

    #[derive(Debug)]
    struct LineEmitter<'a> {
        path: &'a str,
        index: LineIndex,
        errors: HashMap<usize, Vec<(u32, u32, String)>>,
    }

    impl LineEmitter<'_> {
        fn line_col(&self, i: usize) -> LineCol {
            self.index.line_col(TextSize::new(i.try_into().unwrap()))
        }
    }

    impl<'a> Emitter<(&'a str, Range<usize>)> for LineEmitter<'a> {
        fn diagnostic(
            &mut self,
            (path, range): (&'a str, Range<usize>),
            message: impl ToString,
        ) -> impl Diagnostic<(&'a str, Range<usize>)> {
            assert_eq!(path, self.path);
            let start = self.line_col(range.start);
            let end = self.line_col(range.end);
            self.errors
                .entry(u32_to_usize(start.line))
                .or_default()
                .push((
                    start.col,
                    if end.line == start.line {
                        end.col
                    } else {
                        start.col
                    },
                    message.to_string(),
                ));
            IgnoreRelated
        }
    }

    fn differ() -> Differ {
        Box::new(|old, new| {
            similar_asserts::assert_eq!(
                &fs::read_to_string(old).unwrap_or("".to_string()),
                &fs::read_to_string(new).unwrap_or("".to_string()),
                "{}",
                old.display(),
            );
        })
    }

    #[test]
    fn test_errors() {
        let prefix = Path::new("src/typecheck/errors");
        let mut mint = Mint::new(prefix);
        let modules = Modules::new();
        for entry in fs::read_dir(prefix).unwrap() {
            let path = entry.unwrap().path();
            let stripped = path.strip_prefix(prefix).unwrap().to_str().unwrap();
            let source: String = itertools::join(
                fs::read_to_string(&path)
                    .expect(stripped)
                    .lines()
                    .filter(|line| !line.starts_with('#')),
                "\n",
            );
            let tokens = lex(&source).expect(stripped);
            let tree = parse(&tokens).expect(stripped);
            let (module, errors) = typecheck(&source, &tokens, &tree, vec![]);

            let path_str: &str = &path.display().to_string();
            let mut emitter = LineEmitter {
                path: path_str,
                index: LineIndex::new(&source),
                errors: HashMap::new(),
            };
            let full = FullModule {
                source,
                tokens,
                tree,
                imports: vec![],
                module,
            };
            let printer = Printer::new(&modules, &full);
            for error in errors {
                printer.emit_type_error(&mut emitter, path_str, error);
            }

            let mut file = mint
                .new_goldenfile_with_differ(stripped, differ())
                .expect(stripped);
            for (i, line) in full.source.lines().enumerate() {
                writeln!(file, "{}", line).expect(stripped);
                for (start, end, message) in emitter.errors.remove(&i).unwrap_or_default() {
                    writeln!(
                        file,
                        "# {:spaces$}{:^<carets$} {}",
                        "",
                        "",
                        message,
                        spaces = u32_to_usize(start.checked_sub(2).expect(stripped)),
                        carets = u32_to_usize(end - start),
                    )
                    .expect(stripped);
                }
            }
        }
    }
}
