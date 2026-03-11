import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  const config = vscode.workspace.getConfiguration("tiled");

  if (!config.get<boolean>("enable", true)) {
    return;
  }

  const pythonPath = config.get<string>("server.pythonPath", "python3");
  const customServerPath = config.get<string>("server.path", "");
  const extraArgs = config.get<string[]>("server.args", []);

  let serverOptions: ServerOptions;

  if (customServerPath) {
    // Use custom server binary path
    serverOptions = {
      command: customServerPath,
      args: extraArgs,
      transport: TransportKind.stdio,
    };
  } else {
    // Use python -m tiled_server (works if tiled-lsp is pip-installed)
    serverOptions = {
      command: pythonPath,
      args: ["-m", "tiled_server", ...extraArgs],
      transport: TransportKind.stdio,
    };
  }

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "python" }],
    diagnosticCollectionName: "tiled",
    outputChannelName: "TileLang (tiled)",
    middleware: {
      // Only provide tilelang features for files that import tilelang
      provideCompletionItem: async (document, position, context, token, next) => {
        if (!isTileLangFile(document)) {
          return undefined;
        }
        return next(document, position, context, token);
      },
      provideHover: async (document, position, token, next) => {
        if (!isTileLangFile(document)) {
          return undefined;
        }
        return next(document, position, token);
      },
    },
  };

  const traceLevel = config.get<string>("trace.server", "off");

  client = new LanguageClient(
    "tiled",
    "TileLang (tiled)",
    serverOptions,
    clientOptions
  );

  if (traceLevel !== "off") {
    client.setTrace(
      traceLevel === "verbose" ? 2 as any : 1 as any
    );
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("tiled.restart", async () => {
      if (client) {
        await client.restart();
        vscode.window.showInformationMessage("tiled server restarted.");
      }
    })
  );

  await client.start();
}

export async function deactivate(): Promise<void> {
  if (client) {
    await client.stop();
    client = undefined;
  }
}

function isTileLangFile(document: vscode.TextDocument): boolean {
  const text = document.getText();
  return (
    text.includes("import tilelang") ||
    text.includes("from tilelang") ||
    text.includes("tilelang.language")
  );
}
