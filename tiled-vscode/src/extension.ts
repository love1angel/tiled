import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";
import { execFile } from "child_process";

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

  // ── Non-blocking check: warn if tile-lsp is missing, but don't block activation ──
  checkServerInstalled(pythonPath, customServerPath);

  let serverOptions: ServerOptions;

  if (customServerPath) {
    serverOptions = {
      command: customServerPath,
      args: extraArgs,
      transport: TransportKind.stdio,
    };
  } else {
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

  // ── Commands ──────────────────────────────────────────────────────

  context.subscriptions.push(
    vscode.commands.registerCommand("tiled.restart", async () => {
      if (client) {
        await client.restart();
        vscode.window.showInformationMessage("tiled server restarted.");
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("tiled.optimizeKernel", () =>
      runStaticOptimize(pythonPath)
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("tiled.generateKernel", () =>
      pickAndGenerateTemplate(pythonPath)
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("tiled.benchmarkKernel", () =>
      showBenchmarkOptions()
    )
  );

  await client.start();
}

export async function deactivate(): Promise<void> {
  if (client) {
    await client.stop();
    client = undefined;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────

function isTileLangFile(document: vscode.TextDocument): boolean {
  const text = document.getText();
  return (
    text.includes("import tilelang") ||
    text.includes("from tilelang") ||
    text.includes("tilelang.language")
  );
}

/**
 * Run kernel analysis via Copilot's analyze_kernel MCP tool.
 * The old auto_optimize function was removed; analysis now requires tilelang + GPU.
 */
async function runStaticOptimize(_pythonPath: string): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor.");
    return;
  }

  if (!isTileLangFile(editor.document)) {
    vscode.window.showWarningMessage("Current file is not a TileLang kernel.");
    return;
  }

  const fileName = editor.document.fileName.split("/").pop();
  await vscode.commands.executeCommand(
    "workbench.action.chat.open",
    { query: `Use the analyze_kernel tool to analyze and optimize the kernel in ${fileName}` }
  );
}

/**
 * Benchmark requires GPU — show options to the user.
 */
async function showBenchmarkOptions(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor.");
    return;
  }

  if (!isTileLangFile(editor.document)) {
    vscode.window.showWarningMessage("Current file is not a TileLang kernel.");
    return;
  }

  const choice = await vscode.window.showInformationMessage(
    "Benchmarking requires a GPU. How would you like to proceed?",
    "Ask Copilot",
    "Copy CI Command",
    "Cancel"
  );

  if (choice === "Ask Copilot") {
    // Open Copilot chat with a pre-filled prompt
    const fileName = editor.document.fileName.split("/").pop();
    await vscode.commands.executeCommand(
      "workbench.action.chat.open",
      { query: `Use the compile_and_benchmark tool to benchmark the kernel in ${fileName}` }
    );
  } else if (choice === "Copy CI Command") {
    const cmd = `tiled mcp`;
    await vscode.env.clipboard.writeText(cmd);
    vscode.window.showInformationMessage(`Copied MCP server command to clipboard: ${cmd}`);
  }
}

/**
 * Show a quick-pick of available kernel templates, then generate and insert code.
 */
async function pickAndGenerateTemplate(pythonPath: string): Promise<void> {
  const templates = [
    { label: "gemm", description: "Matrix multiplication (GEMM)" },
    { label: "elementwise", description: "Element-wise operation" },
    { label: "reduction", description: "Reduction kernel" },
    { label: "softmax", description: "Softmax kernel" },
    { label: "flash_attention", description: "Flash Attention kernel" },
    { label: "autotune_gemm", description: "Auto-tuning GEMM" },
  ];

  const pick = await vscode.window.showQuickPick(templates, {
    placeHolder: "Select a kernel template",
  });

  if (!pick) {
    return;
  }

  const script = `
import sys
try:
    from tiled_server.knowledge import get_template
    result = get_template(sys.argv[1])
    if result is None:
        print(f"Error: Unknown template '{sys.argv[1]}'", file=sys.stderr)
        sys.exit(1)
    print(result["code"])
except ImportError:
    print("Error: tile-lsp is not installed.", file=sys.stderr)
    print("Install it with: pip install tile-lsp", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
`;

  execFile(
    pythonPath,
    ["-c", script, pick.label],
    { maxBuffer: 512 * 1024 },
    async (error, stdout, stderr) => {
      if (error) {
        vscode.window.showErrorMessage(`Generate failed: ${stderr || error.message}`);
        return;
      }

      const code = stdout;

      const editor = vscode.window.activeTextEditor;
      if (editor) {
        await editor.edit((editBuilder) => {
          editBuilder.insert(editor.selection.active, code);
        });
      } else {
        const doc = await vscode.workspace.openTextDocument({
          content: code,
          language: "python",
        });
        await vscode.window.showTextDocument(doc);
      }
    }
  );
}

// ── Dependency auto-detection ─────────────────────────────────────────

/**
 * Check if tile-lsp is installed and offer to install it if missing.
 * Non-blocking: activation continues regardless.
 */
function checkServerInstalled(
  pythonPath: string,
  customServerPath: string
): void {
  if (customServerPath) {
    return;
  }

  execFile(
    pythonPath,
    ["-c", "import tiled_server; print(tiled_server.__version__)"],
    { timeout: 10000 },
    async (error, stdout, _stderr) => {
      if (!error && stdout.trim()) {
        return;
      }

      const install = "Install tile-lsp";
      const choice = await vscode.window.showWarningMessage(
        "TileLang language server (tile-lsp) is not installed. " +
          "Install it to enable completions, hover docs, and diagnostics.",
        install,
        "Dismiss"
      );

      if (choice === install) {
        const terminal = vscode.window.createTerminal("tile-lsp install");
        terminal.show();
        terminal.sendText(`${pythonPath} -m pip install tile-lsp`);
        vscode.window.showInformationMessage(
          "Installing tile-lsp... Reload the window after installation completes."
        );
      }
    }
  );
}
