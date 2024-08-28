use std::{collections::HashMap, ops::Range, sync::Arc};

use anyhow::anyhow;
use crossbeam_channel::Sender;
use itertools::Itertools;
use line_index::{LineCol, LineIndex};
use lsp_server::{Connection, Message, RequestId, ResponseError};
use lsp_types::{
    notification::{
        DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument,
        Notification, PublishDiagnostics, ShowMessage,
    },
    request::{HoverRequest, Request},
    Diagnostic, DiagnosticRelatedInformation, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams, Hover,
    HoverContents, HoverParams, HoverProviderCapability, Location, MarkupContent, MarkupKind,
    MessageType, Position, PublishDiagnosticsParams, ServerCapabilities, ShowMessageParams,
    TextDocumentPositionParams, TextDocumentSyncCapability, TextDocumentSyncKind,
};
use serde_json::Value;

use crate::{
    compile::{FullModule, GraphImporter, Printer},
    fetch::fetch,
    graph::{Data, Graph, Node, Uri},
    parse::ParseError,
    range,
    typecheck::typecheck,
    util::{self, Emitter},
};

type ResponseResult<T> = Result<T, ResponseError>;

fn notify<N: Notification>(sender: &Sender<Message>, params: N::Params) -> anyhow::Result<()> {
    sender.send(Message::Notification(lsp_server::Notification {
        method: N::METHOD.to_owned(),
        params: serde_json::to_value(params)?,
    }))?;
    Ok(())
}

fn byte_to_lsp(index: &LineIndex, offset: usize) -> Position {
    let LineCol { line, col } = index.line_col(offset.try_into().unwrap());
    Position::new(line, col)
}

fn bytes_to_lsp(index: &LineIndex, range: Range<usize>) -> lsp_types::Range {
    let start = byte_to_lsp(index, range.start);
    let end = byte_to_lsp(index, range.end);
    lsp_types::Range { start, end }
}

fn lsp_to_byte(index: &LineIndex, pos: Position) -> Option<usize> {
    let line = pos.line;
    let col = pos.character;
    Some(index.offset(LineCol { line, col })?.into())
}

#[derive(Debug)]
struct LspEmitter<'a> {
    uri: lsp_types::Uri,
    path: &'a str,
    index: &'a LineIndex,
    diags: Vec<Diagnostic>,
}

impl<'a> Emitter<(&'a str, Range<usize>)> for LspEmitter<'a> {
    fn diagnostic(
        &mut self,
        (path, range): (&'a str, Range<usize>),
        message: impl ToString,
    ) -> impl util::Diagnostic<(&'a str, Range<usize>)> {
        assert_eq!(path, self.path);
        let range = bytes_to_lsp(self.index, range);
        LspDiagnostic {
            emitter: self,
            range,
            message: message.to_string(),
            related: vec![],
        }
    }
}

#[derive(Debug)]
struct LspDiagnostic<'a, 'b> {
    emitter: &'b mut LspEmitter<'a>,
    range: lsp_types::Range,
    message: String,
    related: Vec<DiagnosticRelatedInformation>,
}

impl<'a, 'b> util::Diagnostic<(&'a str, Range<usize>)> for LspDiagnostic<'a, 'b> {
    fn related(mut self, (path, range): (&'a str, Range<usize>), message: impl ToString) -> Self {
        assert_eq!(path, self.emitter.path);
        self.related.push(DiagnosticRelatedInformation {
            location: Location {
                uri: self.emitter.uri.clone(),
                range: bytes_to_lsp(self.emitter.index, range),
            },
            message: message.to_string(),
        });
        self
    }

    fn finish(self) {
        self.emitter.diags.push(Diagnostic {
            range: self.range,
            message: self.message,
            related_information: Some(self.related),
            ..Default::default()
        })
    }
}

#[derive(Debug)]
struct State {
    sender: Sender<Message>,
    graph: Graph,
}

impl State {
    fn new(stdlib: Uri, sender: Sender<Message>) -> Self {
        Self {
            sender,
            graph: Graph::new(stdlib),
        }
    }

    fn notify<N: Notification>(&self, params: N::Params) -> anyhow::Result<()> {
        notify::<N>(&self.sender, params)
    }

    fn exhaust(&mut self) {
        loop {
            let pending = self.graph.pending();
            if pending.is_empty() {
                break;
            }
            for uri in pending {
                if let Ok(text) = fetch(self.graph.stdlib(), &uri) {
                    self.graph.set_text(&uri, text);
                }
            }
        }
        loop {
            let analysis = self.graph.analysis();
            if analysis.is_empty() {
                break;
            }
            for job in analysis {
                let (_, syn, deps) = &job;
                let (module, errs) = typecheck(
                    &syn.src.text,
                    &syn.toks,
                    &syn.tree,
                    deps.iter().map(|(_, dep)| dep.as_ref()).collect(),
                );
                let sem = Arc::new(module);
                self.graph.supply_semantic(job, sem, errs);
            }
        }
    }

    fn diagnose(&self, uri: &Uri, lsp_uri: lsp_types::Uri, node: &Node) -> Vec<Diagnostic> {
        match &node.data {
            Data::Pending => unreachable!(),
            Data::Read { src, err } => {
                let range = bytes_to_lsp(&src.lines, err.byte_range());
                let message = err.message().to_owned();
                vec![Diagnostic::new_simple(range, message)]
            }
            Data::Lexed { src, toks, err } => {
                let id = match *err {
                    ParseError::Expected { id, kinds: _ } => id,
                };
                let range = bytes_to_lsp(&src.lines, toks.get(id).byte_range());
                let message = err.message();
                vec![Diagnostic::new_simple(range, message)]
            }
            Data::Parsed { syn: _ } => vec![],
            Data::Analyzed { syn, sem, errs } => {
                let uri_str = uri.as_str();
                let uris = self.graph.imports(uri).unwrap();
                let full = FullModule {
                    source: &syn.src.text,
                    tokens: &syn.toks,
                    tree: &syn.tree,
                    module: Arc::clone(sem),
                };
                let importer = GraphImporter {
                    graph: &self.graph,
                    uris: &uris,
                };
                let printer = Printer::new(full, importer);
                let mut emitter = LspEmitter {
                    uri: lsp_uri,
                    path: uri_str,
                    index: &syn.src.lines,
                    diags: vec![],
                };
                for &err in errs {
                    printer.emit_type_error(&mut emitter, uri_str, err);
                }
                emitter.diags
            }
        }
    }

    fn diagnose_all(&self) -> anyhow::Result<()> {
        for (uri, node) in self.graph.roots() {
            let lsp_uri = uri
                .to_lsp_uri()
                .map_err(|()| anyhow!("not a valid LSP URI: {}", uri.as_str()))?;
            let diagnostics = self.diagnose(uri, lsp_uri.clone(), node);
            let version = None;
            self.notify::<PublishDiagnostics>(PublishDiagnosticsParams {
                uri: lsp_uri,
                diagnostics,
                version,
            })?;
        }
        Ok(())
    }

    fn hover_success(&self, doc_pos: TextDocumentPositionParams) -> Option<Hover> {
        let uri = Uri::from_lsp_uri(&doc_pos.text_document.uri).ok()?;
        let (syn, sem) = match &self.graph.get(&uri).data {
            Data::Parsed { syn } => (syn, None),
            Data::Analyzed { syn, sem, errs: _ } => (syn, Some(sem)),
            _ => return None,
        };
        let index = &syn.src.lines;
        let offset = lsp_to_byte(index, doc_pos.position)?;
        let (node, bytes) = range::find(&syn.toks, &syn.tree, offset)?;
        let ty = match (sem, node) {
            (Some(_), range::Node::Type(id)) => {
                let _ = id; // TODO: store inferred types from type expressions
                None
            }
            (Some(sem), range::Node::Param(id)) => Some((sem, sem.param(id))),
            (Some(sem), range::Node::Expr(id)) => Some((sem, sem.expr(id))),
            _ => None,
        }
        .map(|(sem, id)| {
            let uris = self.graph.imports(&uri).unwrap();
            let full = FullModule {
                source: &syn.src.text,
                tokens: &syn.toks,
                tree: &syn.tree,
                module: Arc::clone(sem),
            };
            let importer = GraphImporter {
                graph: &self.graph,
                uris: &uris,
            };
            let printer = Printer::new(full, importer);
            format!("{}", printer.ty(sem.val(id).ty))
        });
        let ty_str = match ty.as_ref() {
            Some(s) => s,
            None => "_",
        };
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!("```\n{}\n```", ty_str),
            }),
            range: Some(bytes_to_lsp(index, bytes)),
        })
    }

    fn did_open_text_document(&mut self, params: DidOpenTextDocumentParams) -> anyhow::Result<()> {
        let doc = params.text_document;
        let uri = Uri::from_lsp_uri(&doc.uri).unwrap();
        self.graph.make_root(uri.clone());
        self.graph.set_text(&uri, doc.text);
        self.exhaust();
        self.diagnose_all()
    }

    fn did_change_text_document(
        &mut self,
        params: DidChangeTextDocumentParams,
    ) -> anyhow::Result<()> {
        let doc = params.text_document;
        let uri = Uri::from_lsp_uri(&doc.uri).unwrap();
        let (change,) = params.content_changes.into_iter().collect_tuple().unwrap();
        self.graph.set_text(&uri, change.text);
        self.exhaust();
        self.diagnose_all()
    }

    fn did_save_text_document(&mut self, _: DidSaveTextDocumentParams) -> anyhow::Result<()> {
        Ok(())
    }

    fn did_close_text_document(&mut self, _: DidCloseTextDocumentParams) -> anyhow::Result<()> {
        Ok(())
    }

    fn hover(&self, params: HoverParams) -> ResponseResult<Option<Hover>> {
        Ok(self.hover_success(params.text_document_position_params))
    }
}

type RequestHandler = Box<dyn Fn(&State, RequestId, Value) -> anyhow::Result<()>>;

struct Requests {
    handlers: HashMap<&'static str, RequestHandler>,
}

impl Requests {
    fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    fn with<R: Request>(mut self, f: fn(&State, R::Params) -> ResponseResult<R::Result>) -> Self {
        self.handlers.insert(
            R::METHOD,
            Box::new(move |state, id, params| {
                let (result, error) = match f(state, serde_json::from_value(params)?) {
                    Ok(result) => (Some(serde_json::to_value(result)?), None),
                    Err(error) => (None, Some(error)),
                };
                state.sender.send(Message::Response(lsp_server::Response {
                    id,
                    result,
                    error,
                }))?;
                Ok(())
            }),
        );
        self
    }

    fn handle(&self, state: &State, req: lsp_server::Request) -> anyhow::Result<()> {
        let method: &str = &req.method;
        match self.handlers.get(method) {
            Some(handler) => handler(state, req.id, req.params),
            None => Err(anyhow!("unimplemented request: {method}")),
        }
    }
}

type NotificationHandler = Box<dyn Fn(&mut State, Value) -> anyhow::Result<()>>;

struct Notifications {
    handlers: HashMap<&'static str, NotificationHandler>,
}

impl Notifications {
    fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    fn with<N: Notification>(mut self, f: fn(&mut State, N::Params) -> anyhow::Result<()>) -> Self {
        self.handlers.insert(
            N::METHOD,
            Box::new(move |state, params| f(state, serde_json::from_value(params)?)),
        );
        self
    }

    fn handle(&self, state: &mut State, not: lsp_server::Notification) -> anyhow::Result<()> {
        let method: &str = &not.method;
        match self.handlers.get(method) {
            Some(handler) => handler(state, not.params),
            None => {
                if method.starts_with("$/") {
                    Ok(())
                } else {
                    Err(anyhow!("unimplemented notification: {method}"))
                }
            }
        }
    }
}

fn run(stdlib: Uri, connection: &Connection) -> anyhow::Result<()> {
    let reqs = Requests::new().with::<HoverRequest>(State::hover);
    let nots = Notifications::new()
        .with::<DidChangeTextDocument>(State::did_change_text_document)
        .with::<DidCloseTextDocument>(State::did_close_text_document)
        .with::<DidOpenTextDocument>(State::did_open_text_document)
        .with::<DidSaveTextDocument>(State::did_save_text_document);
    connection.initialize(serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            // TODO: switch to incremental to encourage client to send more frequent updates
            TextDocumentSyncKind::FULL,
        )),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        ..Default::default()
    })?)?;
    let mut state = State::new(stdlib, connection.sender.clone());
    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    break;
                }
                reqs.handle(&state, req)?;
            }
            Message::Response(_) => unreachable!(),
            Message::Notification(not) => nots.handle(&mut state, not)?,
        }
    }
    Ok(())
}

pub fn language_server(stdlib: Uri) -> Result<(), ()> {
    let (connection, io_threads) = Connection::stdio();
    let res = run(stdlib, &connection).map_err(|err| {
        if notify::<ShowMessage>(
            &connection.sender,
            ShowMessageParams {
                typ: MessageType::ERROR,
                message: err.to_string(),
            },
        )
        .is_err()
        {
            eprintln!("language server error: {err}");
        }
        drop(connection); // otherwise thread join below hangs, client thinks server is still up
    });
    match io_threads.join() {
        Ok(()) => res,
        Err(err) => {
            eprintln!("failed to close IO threads: {err}");
            Err(())
        }
    }
}
