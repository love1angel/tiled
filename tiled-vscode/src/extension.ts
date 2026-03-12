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
 * Run auto_optimize on the active file (static analysis, no GPU needed).
 * Detects bugs, adds annotations, optimizes loops — all via code pattern matching.
 */
function runStaticOptimize(pythonPath: string): void {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor.");
    return;
  }

  const doc = editor.document;
  if (!isTileLangFile(doc)) {
    vscode.window.showWarningMessage("Current file is not a TileLang kernel.");
    return;
  }

  const code = doc.getText();
  const channel = vscode.window.createOutputChannel("TileLang Optimization");
  channel.show();
  channel.appendLine("Running auto_optimize (static analysis, no GPU required)...\n");

  const script = `
import json, sys
try:
    from tilelang_mcp.server import auto_optimize
    result = auto_optimize(code=json.loads(sys.argv[1]))
    print(result)
except ImportError:
    print("Error: tilelang-mcp is not installed.", file=sys.stderr)
    print("Install it with: pip install git+https://github.com/tile-ai/tilelang-mcp.git", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
`;

  execFile(
    pythonPath,
    ["-c", script, JSON.stringify(code)],
    { maxBuffer: 1024 * 1024 },
    (error, stdout, stderr) => {
      if (error) {
        channel.appendLine(`Error: ${stderr || error.message}`);
        return;
      }
      channel.appendLine(stdout);
    }
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
    const filePath = vscode.workspace.asRelativePath(editor.document.uri);
    const cmd = `tilelang-mcp ci ${filePath}`;
    await vscode.env.clipboard.writeText(cmd);
    vscode.window.showInformationMessage(`Copied to clipboard: ${cmd}`);
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
    from tilelang_mcp.server import generate_kernel
    result = generate_kernel(template_name=sys.argv[1])
    print(result)
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

      // Extract code block from markdown output
      const codeMatch = stdout.match(/```python\n([\s\S]*?)```/);
      const code = codeMatch ? codeMatch[1] : stdout;

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
