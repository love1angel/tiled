# tiled

**TileLang Language Server, MCP Server & VS Code Extension**

Language intelligence and AI-powered tools for the [TileLang](https://github.com/tile-ai/tilelang) GPU kernel DSL.

## Installation

### From PyPI & VS Code Marketplace (recommended)

**1. Install the language server + MCP server:**

```bash
pip install tile-lsp
```

This installs the `tiled` command — includes both the LSP server and MCP server.

**2. Install the VS Code extension:**

Search for **tiled (TileLang)** in the VS Code Extensions sidebar, or install from the command line:

```bash
code --install-extension tile-ai.vscode-tiled
```

The extension auto-detects if `tile-lsp` is installed. If not, it will prompt you to install it (similar to how clangd extension works).

[![VS Code Marketplace](https://img.shields.io/visual-studio-marketplace/v/tile-ai.vscode-tiled?label=VS%20Code%20Marketplace)](https://marketplace.visualstudio.com/items?itemName=tile-ai.vscode-tiled)
[![PyPI](https://img.shields.io/pypi/v/tile-lsp?label=PyPI)](https://pypi.org/project/tile-lsp/)

**3. Configure (optional):**

The extension works out of the box after installing `tile-lsp`. If needed, add these to your VS Code `settings.json`:

```jsonc
{
  // Use a specific Python interpreter (e.g. a virtualenv where tile-lsp is installed)
  "tiled.server.pythonPath": "/path/to/venv/bin/python3",

  // Or point directly to the tiled binary
  "tiled.server.path": "/path/to/venv/bin/tiled",

  // Enable LSP trace for debugging
  "tiled.trace.server": "verbose"
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `tiled.enable` | `true` | Enable/disable the server |
| `tiled.server.pythonPath` | `python3` | Python interpreter used to run the server |
| `tiled.server.path` | `""` | Custom path to the `tiled` binary (bypasses `pythonPath`) |
| `tiled.server.args` | `[]` | Extra arguments passed to the server |
| `tiled.trace.server` | `off` | LSP trace level (`off`, `messages`, `verbose`) |

### From Source

**Language Server + MCP Server:**

```bash
cd tiled-server
pip install -e .
tiled --help
```

**VS Code Extension:**

```bash
cd tiled-vscode
npm install
npm run bundle
npx @vscode/vsce package
code --install-extension vscode-tiled-1.3.0.vsix
```

## CLI

```bash
# Start LSP server (default, used by VS Code extension automatically)
tiled
tiled lsp
tiled lsp --tcp --port 2087

# Start MCP server (stdio, used by Copilot/AI agents)
tiled mcp

# Check version
tiled --version
```

## Usage

The extension activates automatically when editing Python files that import `tilelang`.

### Completions

Type `T.` to get completions for all TileLang language constructs:

- `T.alloc_shared(...)`, `T.alloc_fragment(...)`, `T.alloc_local(...)`
- `T.Kernel(...)`, `T.Pipelined(...)`, `T.Parallel(...)`
- `T.gemm(...)`, `T.copy(...)`, `T.clear(...)`
- `T.float16`, `T.float32`, `T.bfloat16`, etc.

### Snippets

| Prefix | Description |
|--------|-------------|
| `tl-import` | Import tilelang |
| `tl-gemm` | Full GEMM kernel template |
| `tl-elementwise` | Element-wise kernel template |
| `tl-kernel` | Kernel function skeleton |
| `tl-jit` | JIT wrapper skeleton |
| `tl-attention` | Flash Attention template |
| `tl-pipelined` | Pipelined loop |
| `tl-parallel` | Parallel loop |
| `tl-copy` | Copy operation |
| `tl-alloc-shared` | Shared memory allocation |
| `tl-alloc-fragment` | Fragment allocation |
| `tl-reduce-sum` | Sum reduction |

## Features

### LSP (Language Server)

- **Auto-completion** for `T.*` API (alloc_shared, gemm, Pipelined, etc.)
- **Hover documentation** with signatures and examples
- **Go to Definition** — jump to source for `T.xxx`, `T.XXX.YYY`, `tilelang.xxx`, `tilelang.xxx.yyy`, import aliases, `from tilelang.xxx import ...`, and bare imported symbols
- **Diagnostics** — warns about common mistakes (missing T.clear before T.gemm, etc.)
- **Signature help** when typing function arguments
- **Snippets** — `tl-gemm`, `tl-elementwise`, `tl-kernel`, `tl-attention`, etc.

### MCP Server (AI Agent Tools)

The MCP server provides 9 tools and 4 resources for AI agents (GitHub Copilot, Claude, etc.):

**Tools:**

| Tool | Description | Requires GPU |
|------|------------|:---:|
| `list_apis` | List all API symbols, optionally filtered by category | No |
| `lookup_api` | Look up detailed docs for a specific symbol | No |
| `get_templates` | Get all kernel code templates (gemm, attention, etc.) | No |
| `get_optimization_tips` | Get optimization tips and best practices | No |
| `compile_and_benchmark` | Compile a TileLang kernel and measure performance | Yes |
| `analyze_kernel` | Analyze kernel bottlenecks (FLOPs, roofline, memory) | Yes |
| `autotune` | Auto-tune kernel configurations | Yes |
| `search_examples` | Search TileLang official examples by keyword | No |
| `read_example` | Read source code of a specific example file | No |

**Resources (auto-queried by AI):**

| Resource URI | Description |
|-------------|-------------|
| `tilelang://api/index` | Full API reference (124 symbols, 14 categories) |
| `tilelang://api/{name}` | Detailed docs for a specific API symbol |
| `tilelang://templates` | 6 kernel templates (gemm, elementwise, reduction, softmax, flash_attention, autotune_gemm) |
| `tilelang://optimization-tips` | 10 optimization topics with code examples |
| `tilelang://best-practices` | Best practices for memory, compute, tuning, etc. |

### MCP Setup

The extension registers the MCP server automatically. To configure it manually for your workspace, create `.vscode/mcp.json`:

```json
{
  "servers": {
    "tilelang-mcp": {
      "type": "stdio",
      "command": "tiled",
      "args": ["mcp"]
    }
  }
}
```

> **Tip**: If VS Code can't find `tiled`, use the full path (e.g. `/path/to/venv/bin/tiled`).

Once the MCP server is running, use natural language in Copilot Chat (Agent mode):

- "帮我 benchmark 这个 kernel" → calls `compile_and_benchmark`
- "分析这个 kernel 的性能瓶颈" → calls `analyze_kernel`
- "搜索 flash attention 的示例" → calls `search_examples`
- "autotune this gemm kernel" → calls `autotune`

### VS Code Commands

| Command | Description |
|---------|-------------|
| `TileLang: Restart Language Server` | Restart the LSP server |
| `TileLang: Optimize Current Kernel` | Open Copilot Chat to analyze the current kernel |
| `TileLang: Generate Kernel from Template` | Pick from 6 templates and insert code |
| `TileLang: Benchmark Current Kernel` | Options to benchmark via Copilot or copy command |

### Planned

- [ ] **Find All References** (`textDocument/references`)
- [ ] **Document Symbols** (`textDocument/documentSymbol`) — outline view
- [ ] **Rename Symbol** (`textDocument/rename`)
- [ ] **Code Actions / Quick Fix** (`textDocument/codeAction`) — e.g. auto-insert `T.clear()`
- [ ] **Formatting** (`textDocument/formatting`)
- [ ] **Semantic Tokens** — syntax highlighting for TileLang constructs
- [ ] **Code Lens** — inline performance hints

## Architecture

```
tiled/
├── tiled-server/              # Python package "tile-lsp"
│   ├── tiled_server/
│   │   ├── __main__.py        # CLI: tiled lsp | tiled mcp
│   │   ├── server.py          # LSP server (pygls)
│   │   ├── mcp.py             # MCP server (FastMCP) — 5 tools + 4 resources
│   │   ├── knowledge.py       # Unified knowledge base (shared by LSP & MCP)
│   │   ├── detection.py       # Regex patterns & tilelang file detection
│   │   ├── completion.py      # Completion provider
│   │   ├── hover.py           # Hover provider
│   │   ├── signature.py       # Signature help provider
│   │   ├── definition.py      # Go-to-definition provider
│   │   ├── diagnostics.py     # Diagnostics (common mistake warnings)
│   │   └── utils.py           # Shared helpers
│   ├── tests/
│   │   ├── test_mcp.py        # MCP server tests (33 tests)
│   │   ├── test_knowledge.py
│   │   ├── test_completion.py
│   │   ├── test_definition.py
│   │   ├── test_diagnostics.py
│   │   ├── test_hover.py
│   │   └── ...
│   └── pyproject.toml
└── tiled-vscode/              # VS Code extension "vscode-tiled"
    ├── src/extension.ts       # LSP client, MCP registration, auto-install check
    ├── snippets/tilelang.json
    └── package.json
```

## Development

```bash
# Run all tests (LSP + MCP)
cd tiled-server && python -m pytest tests/ -v

# Run server in TCP mode for debugging
tiled lsp --tcp --port 2087 --log-level debug

# Watch-compile the extension
cd tiled-vscode && npm run watch
```
