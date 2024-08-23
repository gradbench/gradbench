use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use line_index::LineIndex;
use url::Url;

use crate::{
    lex::{lex, LexError, Tokens},
    parse::{self, ParseError},
    typecheck,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Uri(Arc<Url>);

impl Uri {
    fn new(url: Url) -> Self {
        Self(Arc::new(url))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn from_file_path(path: impl AsRef<Path>) -> Result<Self, ()> {
        Ok(Self::new(Url::from_file_path(path)?))
    }

    pub fn from_directory_path(path: impl AsRef<Path>) -> Result<Self, ()> {
        Ok(Self::new(Url::from_directory_path(path)?))
    }

    pub fn to_file_path(&self) -> Result<PathBuf, ()> {
        self.0.to_file_path()
    }

    pub fn from_lsp_uri(uri: &lsp_types::Uri) -> Result<Self, ()> {
        Ok(Self::new(Url::parse(uri.as_str()).map_err(|_| ())?))
    }

    pub fn to_lsp_uri(&self) -> Result<lsp_types::Uri, ()> {
        lsp_types::Uri::from_str(self.0.as_str()).map_err(|_| ())
    }

    fn resolve(&self, stdlib: &Self, name: &str) -> Result<Self, ()> {
        let base = if name.starts_with('.') { self } else { stdlib };
        let url = base.0.join(&format!("{name}.adroit")).map_err(|_| ())?;
        Ok(Self::new(url))
    }
}

#[derive(Debug)]
pub struct Source {
    pub text: String,
    pub lines: LineIndex,
}

impl Source {
    fn new(text: String) -> Self {
        let lines = LineIndex::new(&text);
        Self { text, lines }
    }
}

#[derive(Debug)]
pub struct Syntax {
    pub src: Source,
    pub toks: Tokens,
    pub tree: parse::Module,
}

#[derive(Debug)]
pub enum Data {
    Pending,
    Read {
        src: Source,
        err: LexError,
    },
    Lexed {
        src: Source,
        toks: Tokens,
        err: ParseError,
    },
    Parsed {
        syn: Arc<Syntax>,
    },
    Analyzed {
        syn: Arc<Syntax>,
        sem: Arc<typecheck::Module>,
        errs: Vec<typecheck::TypeError>,
    },
}

impl Data {
    fn new(text: String) -> Self {
        let src = Source::new(text);
        let toks = match lex(&src.text) {
            Ok(toks) => toks,
            Err(err) => return Self::Read { src, err },
        };
        let tree = match parse::parse(&toks) {
            Ok(tree) => tree,
            Err(err) => return Self::Lexed { src, toks, err },
        };
        let syn = Arc::new(Syntax { src, toks, tree });
        Self::Parsed { syn }
    }
}

impl Default for Data {
    fn default() -> Self {
        Self::Pending
    }
}

#[derive(Debug, Default)]
pub struct Node {
    pub root: bool,
    pub imports: HashSet<Uri>,
    pub data: Data,
}

impl Node {
    fn replace(
        &mut self,
        stdlib: &Uri,
        uri: &Uri,
        text: String,
    ) -> Result<Vec<Result<Uri, ()>>, ()> {
        self.data = Data::new(text);
        if !self.root {
            // root nodes are open files; keep prior imports to avoid churn from parse errors
            self.imports.clear();
        }
        if let Data::Parsed { syn } = &self.data {
            Ok(syn
                .tree
                .imports()
                .iter()
                .map(|import| {
                    let token = import.module;
                    let name = syn.toks.get(token).string(&syn.src.text);
                    uri.resolve(stdlib, &name).map(|resolved| {
                        self.imports.insert(resolved.clone());
                        resolved
                    })
                })
                .collect())
        } else {
            Err(())
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    stdlib: Uri,
    nodes: HashMap<Uri, Node>,
}

impl Graph {
    pub fn new(stdlib: Uri) -> Self {
        Self {
            stdlib,
            nodes: HashMap::new(),
        }
    }

    pub fn stdlib(&self) -> &Uri {
        &self.stdlib
    }

    pub fn make_root(&mut self, uri: Uri) {
        self.nodes.entry(uri).or_default().root = true;
    }

    pub fn set_text(&mut self, uri: &Uri, text: String) -> Result<Vec<Result<Uri, ()>>, ()> {
        let node = self.nodes.get_mut(uri).unwrap();
        let imports = node.replace(&self.stdlib, uri, text)?;
        for import in imports.iter().flatten() {
            self.nodes.entry(import.clone()).or_default();
        }
        Ok(imports)
    }

    pub fn get(&self, uri: &Uri) -> &Node {
        &self.nodes[uri]
    }

    pub fn supply_semantic(
        &mut self,
        uri: &Uri,
        sem: Arc<typecheck::Module>,
        errs: Vec<typecheck::TypeError>,
    ) {
        let node = self.nodes.get_mut(uri).unwrap();
        let syn = match &node.data {
            Data::Parsed { syn } => Arc::clone(syn),
            _ => panic!(),
        };
        node.data = Data::Analyzed { syn, sem, errs };
    }
}
