# Changelog

## 1.1.0

### New Features
- **Go to Definition** for `T.xxx` symbols (e.g. `T.gemm`, `T.alloc_shared`, `T.Kernel`)
- **Go to Definition** for `T.xxx.yyy` class member access (e.g. `T.GemmWarpPolicy.FullRow`)
- **Go to Definition** for `tilelang.xxx` top-level symbols (e.g. `@tilelang.jit`)
- **Go to Definition** on import statements — navigate to module files and imported symbol definitions
- **Go to Definition** for bare imported symbol references (e.g. `GemmWarpPolicy.FullRow` after `from ... import GemmWarpPolicy`)

### Improvements
- Dynamic alias detection: works with any `import tilelang.language as XXX` alias, not just `T`

## 1.0.0

- Initial release with completions, hover docs, diagnostics, signature help, and snippets
