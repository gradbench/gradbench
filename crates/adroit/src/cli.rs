use std::{
    fs, io,
    marker::PhantomData,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use ariadne::{Cache, Color, Label, Report, ReportBuilder, ReportKind, Source};
use clap::{Parser, Subcommand};

use crate::{
    compile::{FullModule, Importer, Printer},
    graph::{Data, Graph, Syntax, Uri},
    lsp::language_server,
    parse::ParseError,
    pprint::pprint,
    typecheck::{self, ImportId},
    util::{Diagnostic, Emitter, Id},
};

fn stdlib() -> io::Result<Uri> {
    let dir = dirs::cache_dir()
        .ok_or(io::ErrorKind::NotFound)?
        .join("adroit/modules/");
    Ok(Uri::from_file_path(dir).unwrap())
}

fn path_to_uri(path: &Path) -> io::Result<Uri> {
    Ok(Uri::from_file_path(path.canonicalize()?).unwrap())
}

fn read_uri(uri: &Uri) -> io::Result<String> {
    fs::read_to_string(
        uri.to_file_path()
            .map_err(|_| io::ErrorKind::InvalidInput)?,
    )
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

fn explore<'a>(
    graph: &'a mut Graph,
    uri: &Uri,
) -> Result<(&'a Arc<Syntax>, Vec<Result<Uri, ()>>), ()> {
    let text = read_uri(uri).map_err(|err| eprintln!("{err}"))?;
    let res = graph.set_text(uri, text);
    let node = graph.get(uri);
    match res {
        Ok(imports) => match &node.data {
            Data::Parsed { syn } => Ok((syn, imports)),
            _ => unreachable!(),
        },
        Err(()) => {
            let path = uri.as_str();
            match &node.data {
                Data::Read { src, err } => {
                    AriadneEmitter::new((path, Source::from(&src.text)), "failed to tokenize")
                        .diagnostic((path, err.byte_range()), err.message())
                        .finish();
                    Err(())
                }
                Data::Lexed { src, toks, err } => {
                    let id = match *err {
                        ParseError::Expected { id, kinds: _ } => id,
                    };
                    AriadneEmitter::new((path, Source::from(&src.text)), "failed to parse")
                        .diagnostic((path, toks.get(id).byte_range()), err.message())
                        .finish();
                    Err(())
                }
                _ => unreachable!(),
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GraphImporter<'a> {
    graph: &'a Graph,
    uris: &'a [Uri],
}

impl Importer for GraphImporter<'_> {
    fn import(&self, id: ImportId) -> FullModule {
        match &self.graph.get(&self.uris[id.to_usize()]).data {
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

fn search(graph: &mut Graph, uri: &Uri) -> Result<Arc<typecheck::Module>, ()> {
    let (syntax, results) = explore(graph, uri)?;
    let uris = results.into_iter().collect::<Result<Vec<Uri>, ()>>()?;
    let syn = Arc::clone(syntax);
    let imports = uris
        .iter()
        .map(|import| search(graph, import))
        .collect::<Result<Vec<Arc<typecheck::Module>>, ()>>()?;
    let (module, errs) = typecheck::typecheck(
        &syn.src.text,
        &syn.toks,
        &syn.tree,
        imports.iter().map(|import| import.as_ref()).collect(),
    );
    let sem = Arc::new(module);
    if errs.is_empty() {
        graph.supply_semantic(uri, sem.clone(), errs);
        Ok(sem)
    } else {
        let path = uri.as_str();
        let full = FullModule {
            source: &syn.src.text,
            tokens: &syn.toks,
            tree: &syn.tree,
            module: sem,
        };
        let printer = Printer::new(full, GraphImporter { graph, uris: &uris });
        let mut emitter =
            AriadneEmitter::new((path, Source::from(&syn.src.text)), "failed to typecheck");
        for err in errs {
            printer.emit_type_error(&mut emitter, path, err);
        }
        Err(())
    }
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
            let mut graph = Graph::new(stdlib().map_err(|err| eprintln!("{err}"))?);
            let uri = path_to_uri(&file).map_err(|err| eprintln!("{err}"))?;
            graph.make_root(uri.clone());
            let (syn, _) = explore(&mut graph, &uri)?;
            pprint(&mut io::stdout(), &syn.src.text, &syn.toks, &syn.tree)
                .map_err(|err| eprintln!("{err}"))
        }
        Commands::Json { file } => {
            let mut graph = Graph::new(stdlib().map_err(|err| eprintln!("{err}"))?);
            let uri = path_to_uri(&file).map_err(|err| eprintln!("{err}"))?;
            graph.make_root(uri.clone());
            let sem = search(&mut graph, &uri)?;
            serde_json::to_writer(io::stdout(), sem.as_ref()).map_err(|err| eprintln!("{err}"))?;
            println!();
            Ok(())
        }
        Commands::Lsp => language_server().map_err(|err| eprintln!("{err}")),
    }
}
