"""Diagnostics for TileLang code."""

from __future__ import annotations

import re

from lsprotocol import types as lsp

from .detection import is_tilelang_file


def compute_diagnostics(uri: str, source: str) -> list[lsp.Diagnostic]:
    """Run simple diagnostics on tilelang code."""
    if not is_tilelang_file(source):
        return []

    diagnostics: list[lsp.Diagnostic] = []
    lines = source.split("\n")

    in_prim_func = False
    has_kernel = False
    prim_func_line = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track @T.prim_func
        if stripped == "@T.prim_func":
            in_prim_func = True
            has_kernel = False
            prim_func_line = i

        # Check T.Kernel usage
        if in_prim_func and "T.Kernel" in line:
            has_kernel = True

        # Warn about common mistakes

        # 1. alloc_shared outside Kernel context (heuristic)
        if "T.alloc_shared" in line and not in_prim_func:
            diagnostics.append(lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(line=i, character=line.index("T.alloc_shared")),
                    end=lsp.Position(line=i, character=line.index("T.alloc_shared") + len("T.alloc_shared")),
                ),
                message="T.alloc_shared should be used inside a @T.prim_func with T.Kernel context.",
                severity=lsp.DiagnosticSeverity.Warning,
                source="tiled",
            ))

        # 2. Deprecated or suspicious patterns
        if "T.alloc_buffer" in line and is_tilelang_file(source):
            diagnostics.append(lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(line=i, character=line.index("T.alloc_buffer")),
                    end=lsp.Position(line=i, character=line.index("T.alloc_buffer") + len("T.alloc_buffer")),
                ),
                message="Consider using T.alloc_shared, T.alloc_fragment, or T.alloc_local instead of T.alloc_buffer for clarity.",
                severity=lsp.DiagnosticSeverity.Hint,
                source="tiled",
            ))

        # 3. Missing dtype in Tensor annotation
        tensor_match = re.search(r"T\.Tensor\(\s*\([^)]+\)\s*\)", line)
        if tensor_match:
            diagnostics.append(lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(line=i, character=tensor_match.start()),
                    end=lsp.Position(line=i, character=tensor_match.end()),
                ),
                message="T.Tensor requires a dtype parameter: T.Tensor(shape, dtype)",
                severity=lsp.DiagnosticSeverity.Warning,
                source="tiled",
            ))

        # 4. gemm without clear
        if "T.gemm" in line:
            # Look backward for T.clear
            found_clear = False
            for j in range(max(0, i - 20), i):
                if "T.clear" in lines[j]:
                    found_clear = True
                    break
            if not found_clear:
                col = line.index("T.gemm")
                diagnostics.append(lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(line=i, character=col),
                        end=lsp.Position(line=i, character=col + len("T.gemm")),
                    ),
                    message="T.gemm accumulates into the output buffer. Make sure to T.clear() the accumulator first.",
                    severity=lsp.DiagnosticSeverity.Information,
                    source="tiled",
                ))

        # Reset tracking at function boundaries
        if stripped.startswith("def ") and not in_prim_func:
            in_prim_func = False
            has_kernel = False

    return diagnostics
