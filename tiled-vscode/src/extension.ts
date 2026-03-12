import * as vscode from "vscode";
import { execFileSync } from "child_process";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;
const outputChannel = vscode.window.createOutputChannel("TileLang (tiled)");

/**
 * Find a Python interpreter that has tiled_server installed.
 * Priority: user-configured > python (conda/venv) > python3 > which tiled.
 */
function findPython(configured: string): string | undefined {
  const candidates = configured
    ? [configured]
    : ["python", "python3"];

  for (const py of candidates) {
    try {
      execFileSync(py, ["-c", "import tiled_server"], {
        timeout: 5000,
        stdio: "ignore",
      });
      outputChannel.appendLine(`[tiled] Using Python: ${py}`);
      return py;
    } catch {
      // not available or tiled_server not installed
    }
  }
  return undefined;
}

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  const config = vscode.workspace.getConfiguration("tiled");

  if (!config.get<boolean>("enable", true)) {
    return;
  }

  const configuredPython = config.get<string>("server.pythonPath", "");
  const customServerPath = config.get<string>("server.path", "");
  const extraArgs = config.get<string[]>("server.args", []);

  let serverOptions: ServerOptions;

  if (customServerPath) {
    serverOptions = {
      command: customServerPath,
      args: extraArgs,
      transport: TransportKind.stdio,
    };
  } else {
    const pythonPath = findPython(configuredPython);
    if (!pythonPath) {
      const msg =
        "tiled: Cannot find a Python with tiled_server installed. " +
        'Run "pip install tile-lsp" or set tiled.server.pythonPath.';
      outputChannel.appendLine(`[tiled] ERROR: ${msg}`);
      vscode.window.showErrorMessage(msg);
      return;
    }
    serverOptions = {
      command: pythonPath,
      args: ["-m", "tiled_server", ...extraArgs],
      transport: TransportKind.stdio,
    };
  }

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "python" }],
    diagnosticCollectionName: "tiled",
    outputChannel,
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
