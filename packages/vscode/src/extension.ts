import * as lsp from "vscode-languageclient/node";

let client: lsp.LanguageClient;

export const activate = () => {
  client = new lsp.LanguageClient(
    "adroit",
    "Adroit",
    { command: "adroit", args: ["lsp"] },
    { documentSelector: [{ scheme: "file", language: "adroit" }] },
  );
  client.start();
};

export const deactivate = () => {
  if (client) return client.stop();
};
