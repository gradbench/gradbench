use std::{collections::HashMap, ops::Range};

use anyhow::anyhow;
use crossbeam_channel::Sender;
use itertools::Itertools;
use line_index::{LineCol, LineIndex, TextSize};
use lsp_server::{Connection, Message};
use lsp_types::{
    notification::{
        DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument,
        Notification, PublishDiagnostics, ShowMessage,
    },
    Diagnostic, DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, MessageType, Position, PublishDiagnosticsParams, ServerCapabilities,
    ShowMessageParams, TextDocumentSyncCapability, TextDocumentSyncKind,
};
use serde_json::Value;

use crate::{
    graph::{Data, Graph, Uri},
    parse::ParseError,
};

fn notify<N: Notification>(sender: &Sender<Message>, params: N::Params) -> anyhow::Result<()> {
    sender.send(Message::Notification(lsp_server::Notification {
        method: N::METHOD.to_owned(),
        params: serde_json::to_value(params)?,
    }))?;
    Ok(())
}

fn byte_to_lsp(index: &LineIndex, offset: usize) -> Position {
    let LineCol { line, col } = index.line_col(TextSize::new(offset.try_into().unwrap()));
    Position::new(line, col)
}

fn bytes_to_lsp(index: &LineIndex, range: Range<usize>) -> lsp_types::Range {
    let start = byte_to_lsp(index, range.start);
    let end = byte_to_lsp(index, range.end);
    lsp_types::Range { start, end }
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

    fn update(&mut self, uri: &Uri, text: String) -> Result<(), Vec<Diagnostic>> {
        self.graph.set_text(uri, text);
        let node = self.graph.get(uri);
        match &node.data {
            Data::Read { src, err } => {
                let range = bytes_to_lsp(&src.lines, err.byte_range());
                let message = err.message().to_owned();
                Err(vec![Diagnostic::new_simple(range, message)])
            }
            Data::Lexed { src, toks, err } => {
                let id = match *err {
                    ParseError::Expected { id, kinds: _ } => id,
                };
                let range = bytes_to_lsp(&src.lines, toks.get(id).byte_range());
                let message = err.message();
                Err(vec![Diagnostic::new_simple(range, message)])
            }
            Data::Parsed { syn: _ } => Ok(()),
            _ => unreachable!(),
        }
    }

    fn did_open_text_document(&mut self, params: DidOpenTextDocumentParams) -> anyhow::Result<()> {
        let doc = params.text_document;
        let uri = Uri::from_lsp_uri(&doc.uri).unwrap();
        self.graph.make_root(uri.clone());
        let res = self.update(&uri, doc.text);
        let diagnostics = res.err().unwrap_or_default();
        let version = None;
        self.notify::<PublishDiagnostics>(PublishDiagnosticsParams {
            uri: doc.uri,
            diagnostics,
            version,
        })?;
        Ok(())
    }

    fn did_change_text_document(
        &mut self,
        params: DidChangeTextDocumentParams,
    ) -> anyhow::Result<()> {
        let doc = params.text_document;
        let uri = Uri::from_lsp_uri(&doc.uri).unwrap();
        let (change,) = params.content_changes.into_iter().collect_tuple().unwrap();
        let res = self.update(&uri, change.text);
        let diagnostics = res.err().unwrap_or_default();
        let version = None;
        self.notify::<PublishDiagnostics>(PublishDiagnosticsParams {
            uri: doc.uri,
            diagnostics,
            version,
        })?;
        Ok(())
    }

    fn did_save_text_document(&mut self, _: DidSaveTextDocumentParams) -> anyhow::Result<()> {
        Ok(())
    }

    fn did_close_text_document(&mut self, _: DidCloseTextDocumentParams) -> anyhow::Result<()> {
        Ok(())
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
        ..Default::default()
    })?)?;
    let mut state = State::new(stdlib, connection.sender.clone());
    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    break;
                }
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
