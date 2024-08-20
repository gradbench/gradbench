use std::{collections::HashMap, ops, result};

use itertools::Itertools;
use line_index::{LineCol, LineIndex, TextSize};
use tokio::sync::RwLock;
use tower_lsp::{jsonrpc::Result, lsp_types::*, Client, LanguageServer, LspService, Server};

use crate::{lex, parse};

fn byte_to_lsp(index: &LineIndex, offset: usize) -> Position {
    let LineCol { line, col } = index.line_col(TextSize::new(offset.try_into().unwrap()));
    Position::new(line, col)
}

fn bytes_to_lsp(index: &LineIndex, range: ops::Range<usize>) -> Range {
    let start = byte_to_lsp(index, range.start);
    let end = byte_to_lsp(index, range.end);
    Range { start, end }
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
    docs: HashMap<Url, Doc>,
}

impl State {
    fn update(&mut self, uri: Url, text: String) -> result::Result<(), Vec<Diagnostic>> {
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
}

#[derive(Debug)]
struct Backend {
    client: Client,
    state: RwLock<State>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    // TODO: switch to incremental to encourage client to send more frequent updates
                    TextDocumentSyncKind::FULL,
                )),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let TextDocumentItem { uri, text, .. } = params.text_document;
        let res = self.state.write().await.update(uri.clone(), text);
        let diags = res.err().unwrap_or_default();
        self.client.publish_diagnostics(uri, diags, None).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let VersionedTextDocumentIdentifier { uri, .. } = params.text_document;
        let (change,) = params.content_changes.into_iter().collect_tuple().unwrap();
        let res = self.state.write().await.update(uri.clone(), change.text);
        let diags = res.err().unwrap_or_default();
        self.client.publish_diagnostics(uri, diags, None).await;
    }
}

#[tokio::main]
pub async fn language_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        state: RwLock::new(State {
            docs: HashMap::new(),
        }),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
