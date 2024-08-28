use std::{
    fs, io,
    marker::PhantomData,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use ariadne::{Cache, Color, Label, Report, ReportBuilder, ReportKind, Source};
use clap::{Parser, Subcommand};
use itertools::Itertools;

use crate::{
    compile::{FullModule, Importer, Printer},
    graph::{Analysis, Data, Graph, Syntax, Uri},
    lsp::language_server,
    parse::ParseError,
    pprint::pprint,
    typecheck::{self, ImportId},
    util::{Diagnostic, Emitter, Id},
};

fn builtin(path: &Path) -> Result<&'static str, ()> {
    let name = match path.to_str() {
        Some(string) => string.strip_suffix(".adroit").unwrap(),
        None => {
            eprintln!("module name is not valid Unicode: {}", path.display());
            return Err(());
        }
    };
    match name {
        "array" => Ok(include_str!("modules/array.adroit")),
        "autodiff" => Ok(include_str!("modules/autodiff.adroit")),
        "math" => Ok(include_str!("modules/math.adroit")),
        _ => {
            eprintln!("builtin module does not exist: {name}");
            Err(())
        }
    }
}

fn stdlib() -> Uri {
    let dir = dirs::cache_dir()
        .expect("cache directory should exist")
        .join("adroit/modules");
    Uri::from_directory_path(dir).unwrap()
}

fn rooted_graph(path: PathBuf) -> Result<(Graph, Uri), ()> {
    let mut graph = Graph::new(stdlib());
    let canon = path
        .canonicalize()
        .map_err(|err| eprintln!("error canonicalizing {}: {err}", path.display()))?;
    let uri = Uri::from_file_path(canon).unwrap();
    graph.make_root(uri.clone());
    Ok((graph, uri))
}

#[derive(Debug)]
struct AriadneEmitter<'a, C: Cache<&'a str>> {
    cache: C,
    message: String,
    phantom: PhantomData<&'a ()>,
}

impl<'a, C: Cache<&'a str>> AriadneEmitter<'a, C> {
    fn new(cache: C, message: impl ToString) -> Self {
        Self {
            cache,
            message: message.to_string(),
            phantom: PhantomData,
        }
    }
}

impl<'a, C: Cache<&'a str>> Emitter<(&'a str, Range<usize>)> for AriadneEmitter<'a, C> {
    fn diagnostic(
        &mut self,
        span: (&'a str, Range<usize>),
        message: impl ToString,
    ) -> impl Diagnostic<(&'a str, Range<usize>)> {
        let (path, range) = span.clone();
        AriadneDiagnostic {
            cache: &mut self.cache,
            builder: Report::build(ReportKind::Error, path, range.start)
                .with_message(&self.message)
                .with_label(
                    Label::new(span)
                        .with_color(Color::Red)
                        .with_message(message),
                ),
        }
    }
}

#[derive(Debug)]
struct AriadneDiagnostic<'a, 'b, C: Cache<&'a str>> {
    cache: &'b mut C,
    builder: ReportBuilder<'a, (&'a str, Range<usize>)>,
}

impl<'a, 'b, C: Cache<&'a str>> Diagnostic<(&'a str, Range<usize>)>
    for AriadneDiagnostic<'a, 'b, C>
{
    fn related(mut self, span: (&'a str, Range<usize>), message: impl ToString) -> Self {
        self.builder.add_label(
            Label::new(span)
                .with_color(Color::Blue)
                .with_message(message),
        );
        self
    }

    fn finish(self) {
        self.builder.finish().eprint(self.cache).unwrap();
    }
}

fn read<'a>(graph: &'a mut Graph, uri: &Uri) -> Result<&'a Syntax, ()> {
    let uri_str = uri.as_str();
    let path = uri
        .to_file_path()
        .map_err(|()| eprintln!("not a local file: {uri_str}"))?;
    let stdlib_path = graph.stdlib().to_file_path().unwrap();
    let text = match path.strip_prefix(&stdlib_path) {
        Ok(relative) => {
            let text = builtin(relative)?;
            let parent = path.parent().unwrap();
            fs::create_dir_all(parent)
                .map_err(|err| eprintln!("failed to make directory {}: {err}", parent.display()))?;
            fs::write(&path, text)
                .map_err(|err| eprintln!("failed to write {}: {err}", path.display()))?;
            text.to_owned()
        }
        Err(_) => {
            fs::read_to_string(path).map_err(|err| eprintln!("error reading {uri_str}: {err}"))?
        }
    };
    graph.set_text(uri, text);
    match &graph.get(uri).data {
        Data::Read { src, err } => {
            AriadneEmitter::new((uri_str, Source::from(&src.text)), "failed to tokenize")
                .diagnostic((uri_str, err.byte_range()), err.message())
                .finish();
            Err(())
        }
        Data::Lexed { src, toks, err } => {
            let id = match *err {
                ParseError::Expected { id, kinds: _ } => id,
            };
            AriadneEmitter::new((uri_str, Source::from(&src.text)), "failed to parse")
                .diagnostic((uri_str, toks.get(id).byte_range()), err.message())
                .finish();
            Err(())
        }
        Data::Parsed { syn } => Ok(syn),
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, Debug)]
struct GraphImporter<'a> {
    graph: &'a Graph,
    deps: &'a [(Uri, Arc<typecheck::Module>)],
}

impl Importer for GraphImporter<'_> {
    fn import(&self, id: ImportId) -> FullModule {
        match &self.graph.get(&self.deps[id.to_usize()].0).data {
            Data::Analyzed { syn, sem, errs } => {
                assert!(errs.is_empty());
                FullModule {
                    source: &syn.src.text,
                    tokens: &syn.toks,
                    tree: &syn.tree,
                    module: sem.clone(),
                }
            }
            _ => unreachable!(),
        }
    }
}

fn analyze(graph: &mut Graph, job: Analysis) -> Result<(), ()> {
    let (uri, syn, deps) = &job;
    let (module, errs) = typecheck::typecheck(
        &syn.src.text,
        &syn.toks,
        &syn.tree,
        deps.iter().map(|(_, dep)| dep.as_ref()).collect(),
    );
    let sem = Arc::new(module);
    if errs.is_empty() {
        graph.supply_semantic(job, sem, errs);
        Ok(())
    } else {
        let uri_str = uri.as_str();
        let full = FullModule {
            source: &syn.src.text,
            tokens: &syn.toks,
            tree: &syn.tree,
            module: sem,
        };
        let printer = Printer::new(full, GraphImporter { graph, deps });
        let mut emitter = AriadneEmitter::new(
            (uri_str, Source::from(&syn.src.text)),
            "failed to typecheck",
        );
        for err in errs {
            printer.emit_type_error(&mut emitter, uri_str, err);
        }
        Err(())
    }
}

fn exhaust(graph: &mut Graph) -> Result<(), ()> {
    loop {
        let pending = graph.pending();
        if pending.is_empty() {
            break;
        }
        for uri in pending {
            read(graph, &uri)?;
        }
    }
    loop {
        let analysis = graph.analysis();
        if analysis.is_empty() {
            break;
        }
        for job in analysis {
            analyze(graph, job)?;
        }
    }
    Ok(())
}

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Print the reformatted source code of a module
    Fmt { file: PathBuf },

    /// Print the typed IR of a module as JSON
    Json { file: PathBuf },

    /// Start a language server over stdio
    Lsp,
}

pub fn cli() -> Result<(), ()> {
    match Cli::parse().command {
        Commands::Fmt { file } => {
            let (mut graph, _) = rooted_graph(file)?;
            let (uri,) = graph.pending().into_iter().collect_tuple().unwrap();
            let syn = read(&mut graph, &uri)?;
            pprint(&mut io::stdout(), &syn.src.text, &syn.toks, &syn.tree)
                .map_err(|err| eprintln!("error formatting module: {err}"))
        }
        Commands::Json { file } => {
            let (mut graph, uri) = rooted_graph(file)?;
            exhaust(&mut graph)?;
            let sem = match &graph.get(&uri).data {
                Data::Analyzed { sem, .. } => Ok(sem),
                _ => {
                    eprintln!("cyclic import");
                    Err(())
                }
            }?;
            serde_json::to_writer(io::stdout(), sem.as_ref())
                .map_err(|err| eprintln!("error serializing module: {err}"))?;
            println!();
            Ok(())
        }
        Commands::Lsp => language_server(stdlib()),
    }
}
