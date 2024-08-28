import * as vscode from "vscode";
import * as lsp from "vscode-languageclient/node";

let client: lsp.LanguageClient;

export const activate = (context: vscode.ExtensionContext) => {
  const uri = vscode.Uri.joinPath(context.extensionUri, "bin", "adroit");
  client = new lsp.LanguageClient(
    "adroit",
    "Adroit",
    { command: uri.fsPath, args: ["lsp"] },
    { documentSelector: [{ scheme: "file", language: "adroit" }] },
  );
  client.start();
};

export const deactivate = () => {
  if (client) return client.stop();
};
