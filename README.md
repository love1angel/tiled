# tiled

**TileLang Language Server & VS Code Extension**

Language intelligence for the [TileLang](https://github.com/tile-ai/tilelang) GPU kernel DSL.

## Installation

### From Release (recommended)

Download the latest release from [GitHub Releases](https://github.com/love1angel/tiled/releases).

**1. Install the language server:**

```bash
pip install tile-lsp
```

Or from a downloaded wheel:

```bash
pip install tile_lsp-0.1.0-py3-none-any.whl
```

**2. Install the VS Code extension:**

```bash
code --install-extension vscode-tiled-0.1.0.vsix
```

Or install the `.vsix` manually in VS Code: Extensions sidebar → `···` menu → "Install from VSIX..."

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

**Language Server:**

```bash
cd tiled-server
pip install -e .
tiled --help
```

**VS Code Extension:**

```bash
cd tiled-vscode
npm install
npm run compile
npx @vscode/vsce package
code --install-extension vscode-tiled-0.1.0.vsix
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

- **Auto-completion** for `T.*` API (alloc_shared, gemm, Pipelined, etc.)
- **Hover documentation** with signatures and examples
- **Diagnostics** — warns about common mistakes (missing T.clear before T.gemm, etc.)
- **Signature help** when typing function arguments
- **Snippets** — `tl-gemm`, `tl-elementwise`, `tl-kernel`, `tl-attention`, etc.

## Architecture

```
tiled/
├── tiled-server/              # Python LSP server (pygls)
│   ├── tiled_server/
│   │   ├── __main__.py        # CLI entry point
│   │   ├── server.py          # Server factory & LSP handler wiring
│   │   ├── detection.py       # Regex patterns & tilelang file detection
│   │   ├── completion.py      # Completion provider
│   │   ├── hover.py           # Hover provider
│   │   ├── signature.py       # Signature help provider
│   │   ├── diagnostics.py     # Diagnostics (common mistake warnings)
│   │   ├── knowledge.py       # TileLang API knowledge base
│   │   └── utils.py           # Shared helpers
│   ├── tests/
│   │   ├── conftest.py        # Shared test fixtures
│   │   ├── test_knowledge.py
│   │   ├── test_detection.py
│   │   ├── test_completion.py
│   │   ├── test_diagnostics.py
│   │   ├── test_hover.py
│   │   └── test_server_creation.py
│   └── pyproject.toml
└── tiled-vscode/              # VS Code extension (TypeScript)
    ├── src/extension.ts       # Extension activation, launches tiled server
    ├── snippets/tilelang.json
    └── package.json
```

## Development

```bash
# Run server in TCP mode for debugging
tiled --tcp --port 2087 --log-level debug

# Watch-compile the extension
cd tiled-vscode && npm run watch
```
