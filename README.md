# tiled

**TileLang Language Server & VS Code Extension**

Language intelligence for the [TileLang](https://github.com/tile-ai/tilelang) GPU kernel DSL.

## Features

- **Auto-completion** for `T.*` API (alloc_shared, gemm, Pipelined, etc.)
- **Hover documentation** with signatures and examples
- **Diagnostics** — warns about common mistakes (missing T.clear before T.gemm, etc.)
- **Signature help** when typing function arguments
- **Snippets** — `tl-gemm`, `tl-elementwise`, `tl-kernel`, `tl-attention`, etc.

## Architecture

```
tiled/
├── tiled-server/          # Python LSP server (pygls)
│   ├── tiled_server/
│   │   ├── __main__.py    # CLI entry point
│   │   ├── server.py      # LSP handlers (completion, hover, diagnostics)
│   │   └── knowledge.py   # TileLang API knowledge base
│   └── pyproject.toml
└── tiled-vscode/          # VS Code extension (TypeScript)
    ├── src/extension.ts   # Extension activation, launches tiled server
    ├── snippets/tilelang.json
    └── package.json
```

## Installation

### Language Server

```bash
cd tiled-server
pip install -e .
# Now `tiled` is available as a CLI command
tiled --help
```

### VS Code Extension

```bash
cd tiled-vscode
npm install
npm run compile
# Package as .vsix:
npx @vscode/vsce package
# Install in VS Code:
code --install-extension tiled-0.1.0.vsix
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

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `tiled.enable` | `true` | Enable/disable the server |
| `tiled.server.pythonPath` | `python3` | Python interpreter |
| `tiled.server.path` | `""` | Custom server path |
| `tiled.trace.server` | `off` | LSP trace level |

## Development

```bash
# Run server in TCP mode for debugging
tiled --tcp --port 2087 --log-level debug

# Watch-compile the extension
cd tiled-vscode && npm run watch
```
