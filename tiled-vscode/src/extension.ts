import * as vscode from "vscode";
import { execFileSync } from "child_process";
import * as path from "path";
import * as fs from "fs";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

/**
 * Build a list of Python interpreter candidates to probe.
 */
function pythonCandidates(configured: string): string[] {
  if (configured) {
    return [configured];
  }

  const seen = new Set<string>();
  const candidates: string[] = [];
  const add = (p: string) => {
    if (!seen.has(p)) {
      seen.add(p);
      candidates.push(p);
    }
  };

  // Conda – $CONDA_PREFIX/bin/python (most likely correct)
  const condaPrefix = process.env.CONDA_PREFIX;
  if (condaPrefix) {
    add(path.join(condaPrefix, "bin", "python"));
  }

  // Common conda / pyenv base locations
  const home = process.env.HOME || "";
  for (const base of [
    path.join(home, "miniconda3"),
    path.join(home, "anaconda3"),
    path.join(home, "miniforge3"),
    "/opt/homebrew/Caskroom/miniconda/base",
  ]) {
    const p = path.join(base, "bin", "python");
    if (fs.existsSync(p)) {
      add(p);
    }
  }

  // PATH-based fallbacks
  add("python");
  add("python3");

  return candidates;
}

/**
 * Test if a Python interpreter has tiled_server installed.
 */
function hasTiledServer(py: string): boolean {
  try {
    execFileSync(py, ["-c", "import tiled_server"], {
      timeout: 3000,
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Find a Python interpreter that has tiled_server installed.
 */
function findPython(
  configured: string,
  outputChannel: vscode.OutputChannel
): string | undefined {
  const candidates = pythonCandidates(configured);

  for (const py of candidates) {
    if (hasTiledServer(py)) {
      outputChannel.appendLine(`[tiled] Using Python: ${py}`);
      return py;
    }
  }

  outputChannel.appendLine(
    `[tiled] Could not find tiled_server. Tried: ${candidates.join(", ")}`
  );
  return undefined;
}

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  const outputChannel = vscode.window.createOutputChannel("TileLang (tiled)");
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
    const pythonPath = findPython(configuredPython, outputChannel);
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

  try {
    await client.start();
    outputChannel.appendLine("[tiled] Server started successfully.");
  } catch (err) {
    outputChannel.appendLine(`[tiled] Server failed to start: ${err}`);
    outputChannel.show(true);
  }
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
