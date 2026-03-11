"""Hover provider for TileLang code."""

from __future__ import annotations

import re
from typing import Optional

from lsprotocol import types as lsp

from .detection import is_tilelang_file
from .knowledge import lookup_symbol


def build_hover(document, position: lsp.Position) -> Optional[lsp.Hover]:
    """Build hover information for TileLang symbols."""
    source = document.source

    if not is_tilelang_file(source):
        return None

    pos = position
    lines = document.lines
    if pos.line >= len(lines):
        return None

    line = lines[pos.line]

    # Find the word under cursor — look for T.xxx pattern
    for m in re.finditer(r"\bT\.(\w+)", line):
        start, end = m.start(1), m.end(1)
        if start <= pos.character <= end:
            sym = lookup_symbol(m.group(1))
            if sym:
                return lsp.Hover(
                    contents=lsp.MarkupContent(
                        kind=lsp.MarkupKind.Markdown,
                        value=f"**{sym.detail}**\n\n{sym.documentation}",
                    ),
                    range=lsp.Range(
                        start=lsp.Position(line=pos.line, character=m.start()),
                        end=lsp.Position(line=pos.line, character=m.end()),
                    ),
                )

    # Look for tilelang.xxx pattern
    for m in re.finditer(r"\btilelang\.(\w+)", line):
        start, end = m.start(1), m.end(1)
        if start <= pos.character <= end:
            sym = lookup_symbol(m.group(1))
            if sym:
                return lsp.Hover(
                    contents=lsp.MarkupContent(
                        kind=lsp.MarkupKind.Markdown,
                        value=f"**{sym.detail}**\n\n{sym.documentation}",
                    ),
                    range=lsp.Range(
                        start=lsp.Position(line=pos.line, character=m.start()),
                        end=lsp.Position(line=pos.line, character=m.end()),
                    ),
                )

    return None
