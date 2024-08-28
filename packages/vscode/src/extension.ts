import * as process from "process";
import * as vscode from "vscode";
import * as lsp from "vscode-languageclient/node";

let client: lsp.LanguageClient;

export const activate = (context: vscode.ExtensionContext) => {
  const uri = context.extensionUri;
  const ext = process.platform === "win32" ? ".exe" : "";
  const command = vscode.Uri.joinPath(uri, "bin", `adroit${ext}`).fsPath;
  client = new lsp.LanguageClient(
    "adroit",
    "Adroit",
    { command, args: ["lsp"] },
    { documentSelector: [{ scheme: "file", language: "adroit" }] },
  );
  client.start();
};

export const deactivate = () => {
  if (client) return client.stop();
};
