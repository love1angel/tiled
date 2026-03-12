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
      runMcpOnActiveFile(pythonPath, "optimize")
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("tiled.generateKernel", () =>
      pickAndGenerateTemplate(pythonPath)
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("tiled.benchmarkKernel", () =>
      runMcpOnActiveFile(pythonPath, "benchmark")
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
 * Run tilelang-mcp auto_optimize or compile_and_benchmark on the active file,
 * showing results in an output channel.
 */
function runMcpOnActiveFile(pythonPath: string, mode: "optimize" | "benchmark"): void {
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

  const toolName = mode === "optimize" ? "auto_optimize" : "compile_and_benchmark";
  channel.appendLine(`Running ${toolName}...\n`);

  // Call tilelang-mcp via python -m tilelang_mcp.run_tool
  const script = `
import json, sys
sys.path.insert(0, "")
try:
    from tilelang_mcp.server import ${toolName}
    result = ${toolName}(code=json.loads(sys.argv[1]))
    print(result)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
`;

  const child = execFile(
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
