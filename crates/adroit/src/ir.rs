use std::collections::HashMap;

use indexmap::IndexSet;

use crate::{
    graph::{Data, Graph, Syntax, Uri},
    parse, typecheck,
    util::{u32_to_usize, Id},
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExprId {
    pub index: u32,
}

impl Id for ExprId {
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Type {
    Unit,
    I32,
    F64,
    Prod { fst: TypeId, snd: TypeId },
    Sum { left: TypeId, right: TypeId },
    Func { dom: TypeId, cod: TypeId },
    Array { elem: TypeId },
    Cell { inner: TypeId },
}

#[derive(Clone, Copy, Debug)]
pub enum Expr {
    Var {
        id: VarId,
    },
    Unit,
    Bool {
        value: bool,
    },
    I32 {
        value: i32,
    },
    F64 {
        value: f64,
    },
    Pair {
        fst: ExprId,
        snd: ExprId,
    },
    Fst {
        value: ExprId,
    },
    Snd {
        value: ExprId,
    },
    Inl {
        value: ExprId,
    },
    Inr {
        value: ExprId,
    },
    Array {
        len: ExprId,
        elem: ExprId,
    },
    Index {
        array: ExprId,
        index: ExprId,
    },
    Cell {
        value: ExprId,
    },
    Get {
        cell: ExprId,
    },
    Set {
        cell: ExprId,
        value: ExprId,
    },
    Call {
        func: ExprId,
        arg: ExprId,
    },
    Tail {
        func: ExprId,
        arg: ExprId,
    },
    Let {
        bind: VarId,
        val: ExprId,
        body: ExprId,
    },
    Lambda {
        param: VarId,
        body: ExprId,
    },
    If {
        cond: ExprId,
        yes: ExprId,
        no: ExprId,
    },
    Case {
        value: ExprId,
        inl: ExprId,
        inr: ExprId,
    },
}

#[derive(Debug)]
struct Defs {
    vars: Box<[VarId]>,
    exprs: Box<[ExprId]>,
}

#[derive(Debug)]
struct Lowerer<'a> {
    graph: &'a Graph,
    types: IndexSet<Type>,
    var_types: Vec<TypeId>,
    expr_types: Vec<TypeId>,
    exprs: Vec<Expr>,
    modules: HashMap<Uri, Defs>,
}

#[derive(Clone, Copy, Debug)]
struct Context<'a> {
    syn: &'a Syntax,
    sem: &'a typecheck::Module,
    types: &'a [Option<TypeId>],
    vars: &'a [VarId],
}

impl<'a> Lowerer<'a> {
    fn ty(&mut self, ty: Type) -> TypeId {
        let (i, _) = self.types.insert_full(ty);
        TypeId::from_usize(i).unwrap()
    }

    fn var(&mut self, t: TypeId) -> VarId {
        let i = self.var_types.len();
        self.var_types.push(t);
        VarId::from_usize(i).unwrap()
    }

    fn lower_ty(
        &mut self,
        raw_types: &typecheck::Types,
        types: &[Option<TypeId>],
        t: typecheck::TypeId,
    ) -> Option<TypeId> {
        match raw_types.get(t) {
            typecheck::Type::Unknown { .. }
            | typecheck::Type::Scalar { .. }
            | typecheck::Type::Vector { .. } => unreachable!(),
            typecheck::Type::Fragment
            | typecheck::Type::Var { .. }
            | typecheck::Type::Poly { .. } => None,
            typecheck::Type::Unit => Some(self.ty(Type::Unit)),
            typecheck::Type::Int => Some(self.ty(Type::I32)),
            typecheck::Type::Float => Some(self.ty(Type::F64)),
            typecheck::Type::Prod { fst, snd } => Some(self.ty(Type::Prod {
                fst: types[fst.to_usize()].unwrap(),
                snd: types[snd.to_usize()].unwrap(),
            })),
            typecheck::Type::Sum { left, right } => Some(self.ty(Type::Sum {
                left: types[left.to_usize()].unwrap(),
                right: types[right.to_usize()].unwrap(),
            })),
            typecheck::Type::Array { index: _, elem } => Some(self.ty(Type::Array {
                elem: types[elem.to_usize()].unwrap(),
            })),
            typecheck::Type::Record {
                name: _,
                field,
                rest,
            } => Some(self.ty(Type::Prod {
                fst: types[field.to_usize()].unwrap(),
                snd: types[rest.to_usize()].unwrap(),
            })),
            typecheck::Type::End => Some(self.ty(Type::Unit)),
            typecheck::Type::Func { dom, cod } => Some(self.ty(Type::Func {
                dom: types[dom.to_usize()].unwrap(),
                cod: types[cod.to_usize()].unwrap(),
            })),
        }
    }

    fn lower_expr(&mut self, ctx: Context, id: parse::ExprId) -> ExprId {
        let typecheck::Val { ty, src } = ctx.sem.val(ctx.sem.expr(id));
        match ctx.syn.tree.expr(id) {
            parse::Expr::Paren { inner } => self.lower_expr(ctx, inner),
            parse::Expr::Name { name: _ } => todo!(),
            parse::Expr::Undefined { token: _ } => panic!(),
            parse::Expr::Unit { open: _, close: _ } => todo!(),
            parse::Expr::Number { val } => todo!(),
            parse::Expr::Pair { fst, snd } => todo!(),
            parse::Expr::Record { name, field, rest } => todo!(),
            parse::Expr::End { open, close } => todo!(),
            parse::Expr::Elem { array, index } => todo!(),
            parse::Expr::Inst { val, ty } => todo!(),
            parse::Expr::Apply { func, arg } => todo!(),
            parse::Expr::Map { func, arg } => todo!(),
            parse::Expr::Let { param, val, body } => todo!(),
            parse::Expr::Index { name, val, body } => todo!(),
            parse::Expr::Unary { op, arg } => todo!(),
            parse::Expr::Binary { lhs, op, rhs } => todo!(),
            parse::Expr::Lambda { param, ty, body } => todo!(),
        }
    }

    fn lower_module(&mut self, uri: &Uri) {
        if self.modules.contains_key(uri) {
            return;
        }
        match &self.graph.get(uri).data {
            Data::Analyzed { syn, sem, errs } => {
                assert!(errs.is_empty());
                let mut types = vec![];
                let raw_types = sem.types();
                for i in 0..raw_types.len() {
                    let id = typecheck::TypeId::from_usize(i).unwrap();
                    types.push(self.lower_ty(raw_types, &types, id));
                }
                let vars = syn
                    .tree
                    .defs()
                    .iter()
                    .enumerate()
                    .map(|(i, _)| self.var(types[i].unwrap()))
                    .collect::<Vec<VarId>>()
                    .into_boxed_slice();
                let ctx = Context {
                    syn,
                    sem,
                    types: &types,
                    vars: &vars,
                };
                let exprs = syn
                    .tree
                    .exprs()
                    .iter()
                    .enumerate()
                    .map(|(i, _)| self.lower_expr(ctx, parse::ExprId::from_usize(i).unwrap()))
                    .collect::<Vec<ExprId>>()
                    .into_boxed_slice();
                self.modules.insert(uri.clone(), Defs { vars, exprs });
            }
            _ => panic!(),
        }
    }
}

pub fn lower(graph: &Graph) {
    let mut lowerer = Lowerer {
        graph,
        types: IndexSet::new(),
        var_types: vec![],
        expr_types: vec![],
        exprs: vec![],
        modules: HashMap::new(),
    };
    for (uri, _) in graph.roots() {
        lowerer.lower_module(uri);
    }
}
