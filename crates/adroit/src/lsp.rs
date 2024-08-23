use std::{collections::HashMap, ops::Range};

use anyhow::anyhow;
use crossbeam_channel::Sender;
use itertools::Itertools;
use line_index::{LineCol, LineIndex, TextSize};
use lsp_server::{Connection, Message};
use lsp_types::{
    notification::{
        DidChangeTextDocument, DidOpenTextDocument, DidSaveTextDocument, Notification,
        PublishDiagnostics,
    },
    Diagnostic, DidChangeTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
    Position, PublishDiagnosticsParams, ServerCapabilities, TextDocumentItem,
    TextDocumentSyncCapability, TextDocumentSyncKind, Uri, VersionedTextDocumentIdentifier,
};
use serde_json::Value;

use crate::{lex, parse};

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
struct Doc {
    text: String,
    index: LineIndex,
    tokens: lex::Tokens,
    tree: parse::Module,
}

#[derive(Debug)]
struct State {
    sender: Sender<Message>,
    docs: HashMap<Uri, Doc>,
}

impl State {
    fn new(sender: Sender<Message>) -> Self {
        Self {
            sender,
            docs: HashMap::new(),
        }
    }

    fn notify<N: Notification>(&self, params: N::Params) -> anyhow::Result<()> {
        self.sender
            .send(Message::Notification(lsp_server::Notification {
                method: N::METHOD.to_owned(),
                params: serde_json::to_value(params)?,
            }))?;
        Ok(())
    }

    fn update(&mut self, uri: Uri, text: String) -> Result<(), Vec<Diagnostic>> {
        let index = LineIndex::new(&text);
        let tokens = lex::lex(&text).map_err(|err| {
            let range = bytes_to_lsp(&index, err.byte_range());
            let message = err.message().to_owned();
            vec![Diagnostic::new_simple(range, message)]
        })?;
        let tree = parse::parse(&tokens).map_err(|err| {
            let id = match err {
                parse::ParseError::Expected { id, kinds: _ } => id,
            };
            let range = bytes_to_lsp(&index, tokens.get(id).byte_range());
            let message = err.message();
            vec![Diagnostic::new_simple(range, message)]
        })?;
        let doc = Doc {
            text,
            index,
            tokens,
            tree,
        };
        self.docs.insert(uri, doc);
        Ok(())
    }

    fn did_open_text_document(&mut self, params: DidOpenTextDocumentParams) -> anyhow::Result<()> {
        let TextDocumentItem { uri, text, .. } = params.text_document;
        let res = self.update(uri.clone(), text);
        let diagnostics = res.err().unwrap_or_default();
        let version = None;
        self.notify::<PublishDiagnostics>(PublishDiagnosticsParams {
            uri,
            diagnostics,
            version,
        })?;
        Ok(())
    }

    fn did_change_text_document(
        &mut self,
        params: DidChangeTextDocumentParams,
    ) -> anyhow::Result<()> {
        let VersionedTextDocumentIdentifier { uri, .. } = params.text_document;
        let (change,) = params.content_changes.into_iter().collect_tuple().unwrap();
        let res = self.update(uri.clone(), change.text);
        let diagnostics = res.err().unwrap_or_default();
        let version = None;
        self.notify::<PublishDiagnostics>(PublishDiagnosticsParams {
            uri,
            diagnostics,
            version,
        })?;
        Ok(())
    }

    fn did_save_text_document(&mut self, _: DidSaveTextDocumentParams) -> anyhow::Result<()> {
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
        self.handlers
            .get(method)
            .ok_or_else(|| anyhow!("unimplemented notification: {method}"))?(
            state, not.params
        )
    }
}

fn run(mut connection: Connection) -> anyhow::Result<()> {
    let nots = Notifications::new()
        .with::<DidChangeTextDocument>(State::did_change_text_document)
        .with::<DidOpenTextDocument>(State::did_open_text_document)
        .with::<DidSaveTextDocument>(State::did_save_text_document);
    connection.initialize(serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            // TODO: switch to incremental to encourage client to send more frequent updates
            TextDocumentSyncKind::FULL,
        )),
        ..Default::default()
    })?)?;
    let mut state = State::new(connection.sender);
    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                connection.sender = state.sender;
                if connection.handle_shutdown(&req)? {
                    break;
                }
                state.sender = connection.sender;
            }
            Message::Response(_) => unreachable!(),
            Message::Notification(not) => nots.handle(&mut state, not)?,
        }
    }
    Ok(())
}

pub fn language_server() -> Result<(), ()> {
    let (connection, io_threads) = Connection::stdio();
    match run(connection) {
        Ok(()) => io_threads.join().map_err(|err| eprintln!("{err}")),
        Err(err) => {
            eprintln!("{err}");
            Err(())
        }
    }
}
