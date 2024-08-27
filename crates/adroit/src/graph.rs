use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    mem::{replace, take},
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
        dirty: bool,
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
    /// Whether or not this node is a root.
    ///
    /// When running a module as a script, that module is a root node, and all the others are not.
    /// When running the language server, being a root node is the same as being an open file.
    ///
    /// Root nodes are always kept alive. They also keep alive all their (transitive) dependencies.
    /// Importantly, while for most nodes the dependencies are just the set of imported URIs, for
    /// root nodes, the dependencies are a superset of that. Specifically, the set of dependencies
    /// of a root node can only grow until that node is no longer a root. We do this because we
    /// expect root nodes to undergo frequent edits that transition them between syntactically valid
    /// and invalid states, and we don't want to drop and then rebuild the entire module graph every
    /// time the user types two characters.
    pub root: bool,

    /// The set of URIs that this node keeps alive.
    ///
    /// If the node is not a root, this is just the set of resolved URIs from the imports in the
    /// parsed AST, or empty if the source text is not read or the text was not successfully parsed.
    /// If the node is a root, this is a superset of that, also containing any other URIs that were
    /// present as imports since this node was originally made a root. When a node is transitioned
    /// from a root to a non-root, this is reset to the current set of imported URIs.
    pub dependencies: HashSet<Uri>,

    /// The set of URIs that import this node.
    ///
    /// Unlike the dependencies, this does not depend on whether any nodes are roots or not. In
    /// particular, a different node could include this node's URI in its set of dependencies
    /// because that other node is a root, but this node could not include the other node as a
    /// dependent because the other node no longer actually imports this one.
    pub dependents: HashSet<Uri>,

    /// The state and contents of this node.
    pub data: Data,
}

#[derive(Debug)]
pub enum Job {
    Read {
        uri: Uri,
    },
    Typecheck {
        uri: Uri,
        syn: Arc<Syntax>,
        deps: Vec<(Uri, Arc<typecheck::Module>)>,
    },
}

#[derive(Debug)]
struct Jobs {
    jobs: HashMap<Uri, Job>,
}

impl Jobs {
    fn new() -> Self {
        Self {
            jobs: HashMap::new(),
        }
    }

    fn make(&mut self, uri: Uri, job: Job) {
        assert!(self.jobs.insert(uri, job).is_none());
    }
}

#[derive(Debug)]
pub struct Graph {
    stdlib: Uri,
    nodes: HashMap<Uri, Node>,
    jobs: Jobs,
}

impl Graph {
    pub fn new(stdlib: Uri) -> Self {
        Self {
            stdlib,
            nodes: HashMap::new(),
            jobs: Jobs::new(),
        }
    }

    pub fn stdlib(&self) -> &Uri {
        &self.stdlib
    }

    pub fn get(&self, uri: &Uri) -> &Node {
        &self.nodes[uri]
    }

    fn make_node(&mut self, uri: Uri) -> &mut Node {
        match self.nodes.entry(uri.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                self.jobs.make(uri.clone(), Job::Read { uri });
                entry.insert(Default::default())
            }
        }
    }

    fn replace_text(&mut self, uri: &Uri, text: String) {
        // currently we never delete nodes, so this should always be fine; in future we should
        // consider just returning early if there is no node with this URI, which could maybe happen
        // if someone completes a job to read a file after that file is no longer needed
        let node = self.nodes.get_mut(uri).unwrap();
        let mut kill = if node.root {
            HashSet::new()
        } else {
            take(&mut node.dependencies)
        };
        let mut succs: Vec<Uri> = node.dependents.iter().cloned().collect();
        node.data = Data::new(text);
        if let Data::Parsed { syn } = &node.data {
            let syn = Arc::clone(syn);
            let imports: Vec<Option<Uri>> = syn
                .tree
                .imports()
                .iter()
                .map(|import| {
                    let name = syn.toks.get(import.module).string(&syn.src.text);
                    match uri.resolve(&self.stdlib, &name) {
                        Ok(resolved) => {
                            kill.remove(&resolved);
                            node.dependencies.insert(resolved.clone());
                            Some(resolved)
                        }
                        Err(()) => None,
                    }
                })
                .collect();
            let res: Result<Vec<(Uri, Arc<typecheck::Module>)>, ()> = imports
                .into_iter()
                .map(|opt| {
                    if let Some(import) = opt {
                        let dep = self.make_node(import.clone());
                        dep.dependents.insert(uri.clone());
                        if let Data::Analyzed { sem, dirty, .. } = &dep.data {
                            if !dirty {
                                return Ok((import, sem.clone()));
                            }
                        }
                    }
                    return Err(());
                })
                .collect();
            if let Ok(deps) = res {
                let job = Job::Typecheck {
                    uri: uri.clone(),
                    syn,
                    deps,
                };
                self.jobs.make(uri.clone(), job);
            }
        }
        for pred in kill {
            self.nodes.get_mut(&pred).unwrap().dependents.remove(uri);
            // TODO: delete all nodes that roots don't depend on
        }
        while let Some(succ) = succs.pop() {
            let node = self.nodes.get_mut(&succ).unwrap();
            if let Data::Analyzed { dirty, .. } = &mut node.data {
                if !replace(dirty, true) {
                    succs.extend(node.dependents.iter().cloned());
                }
            }
        }
    }

    pub fn make_root(&mut self, uri: Uri) {
        self.make_node(uri.clone()).root = true;
    }

    pub fn set_text(&mut self, job: Job, text: String) {
        let uri = match job {
            Job::Read { uri } => uri,
            _ => panic!(),
        };
        self.replace_text(&uri, text)
    }

    pub fn change_text(&mut self, uri: &Uri, text: String) {
        self.replace_text(uri, text)
    }

    pub fn supply_semantic(
        &mut self,
        job: Job,
        sem: Arc<typecheck::Module>,
        errs: Vec<typecheck::TypeError>,
    ) {
        let (uri, syn, deps) = match job {
            Job::Typecheck { uri, syn, deps } => (uri, syn, deps),
            _ => panic!(),
        };
        let node = match self.nodes.get(&uri) {
            Some(node) => node,
            None => return, // node was removed from the graph, no longer needed
        };
        let current = match &node.data {
            Data::Parsed { syn } => syn,
            Data::Analyzed { syn, .. } => syn,
            _ => return, // lex or parse error so text must have changed; the result is outdated
        };
        if !Arc::ptr_eq(&syn, current) {
            return; // the text changed, so the typechecking result is outdated
        }
        if !deps.into_iter().all(|(import, before)| {
            if let Data::Analyzed { sem, dirty, .. } = &self.get(&import).data {
                if !dirty {
                    return Arc::ptr_eq(&before, sem);
                }
            }
            return false;
        }) {
            return; // dependencies changed, so the typechecking result is outdated
        }
        let node = self.nodes.get_mut(&uri).unwrap();
        let succs: Vec<Uri> = node.dependents.iter().cloned().collect();
        node.data = Data::Analyzed {
            syn,
            sem,
            errs,
            dirty: false,
        };
        for _ in succs {
            // TODO: make jobs for all dependents that have no dirty dependencies
        }
    }
}
