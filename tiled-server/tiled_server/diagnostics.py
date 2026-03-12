"""Diagnostics for TileLang code."""

from __future__ import annotations

import re

from lsprotocol import types as lsp

from .detection import is_tilelang_file

_RE_TRIPLE_QUOTE = re.compile(r'("""|\'\'\')')
_RE_STRING_OR_COMMENT = re.compile(
    r'#.*$|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.MULTILINE
)


def _strip_strings_and_comments(line: str) -> str:
    """Replace string literals and comments with spaces (preserving positions)."""
    def _blank(m: re.Match) -> str:
        return " " * len(m.group(0))
    return _RE_STRING_OR_COMMENT.sub(_blank, line)


def _string_lines(source: str) -> set[int]:
    """Return the set of line numbers that fall inside triple-quoted strings."""
    inside: set[int] = set()
    in_string = False
    quote_char = ""
    for i, line in enumerate(source.split("\n")):
        had_toggle = False
        for m in _RE_TRIPLE_QUOTE.finditer(line):
            if not in_string:
                in_string = True
                quote_char = m.group(1)
                had_toggle = True
            elif m.group(1) == quote_char:
                in_string = False
                had_toggle = True
        # Mark opening, closing, and interior lines
        if in_string or had_toggle:
            inside.add(i)
    return inside


def compute_diagnostics(uri: str, source: str) -> list[lsp.Diagnostic]:
    """Run simple diagnostics on tilelang code."""
    if not is_tilelang_file(source):
        return []

    diagnostics: list[lsp.Diagnostic] = []
    lines = source.split("\n")
    skip = _string_lines(source)

    in_prim_func = False
    has_kernel = False
    prim_func_line = 0

    for i, line in enumerate(lines):
        if i in skip:
            continue
        # Strip string literals and comments so we don't flag code inside them
        code = _strip_strings_and_comments(line)
        stripped = code.strip()

        # Track @T.prim_func
        if stripped == "@T.prim_func":
            in_prim_func = True
            has_kernel = False
            prim_func_line = i

        # Check T.Kernel usage
        if in_prim_func and "T.Kernel" in code:
            has_kernel = True

        # Warn about common mistakes

        # 1. alloc_shared outside Kernel context (heuristic)
        if "T.alloc_shared" in code and not in_prim_func:
            diagnostics.append(lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(line=i, character=code.index("T.alloc_shared")),
                    end=lsp.Position(line=i, character=code.index("T.alloc_shared") + len("T.alloc_shared")),
                ),
                message="T.alloc_shared should be used inside a @T.prim_func with T.Kernel context.",
                severity=lsp.DiagnosticSeverity.Warning,
                source="tiled",
            ))

        # 2. Deprecated or suspicious patterns
        if "T.alloc_buffer" in code and is_tilelang_file(source):
            diagnostics.append(lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(line=i, character=code.index("T.alloc_buffer")),
                    end=lsp.Position(line=i, character=code.index("T.alloc_buffer") + len("T.alloc_buffer")),
                ),
                message="Consider using T.alloc_shared, T.alloc_fragment, or T.alloc_local instead of T.alloc_buffer for clarity.",
                severity=lsp.DiagnosticSeverity.Hint,
                source="tiled",
            ))

        # 3. Missing dtype in Tensor annotation
        tensor_match = re.search(r"T\.Tensor\(\s*\([^)]+\)\s*\)", code)
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
        if "T.gemm" in code:
            # Look backward for T.clear
            found_clear = False
            for j in range(max(0, i - 20), i):
                if "T.clear" in _strip_strings_and_comments(lines[j]):
                    found_clear = True
                    break
            if not found_clear:
                col = code.index("T.gemm")
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
