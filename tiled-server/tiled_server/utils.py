"""Utility helpers for the tiled language server."""

from __future__ import annotations

import re
from typing import Optional

from lsprotocol import types as lsp

from .detection import _RE_T_DOT


def get_line_before_cursor(document, position: lsp.Position) -> str:
    """Get text from the start of the line to the cursor position."""
    lines = document.lines
    if position.line >= len(lines):
        return ""
    line = lines[position.line]
    return line[: position.character]


def get_word_at_position(document, position: lsp.Position) -> Optional[str]:
    """Get the word (including T. prefix) at the cursor position."""
    line_text = get_line_before_cursor(document, position)

    # Check for T.something
    m = _RE_T_DOT.search(line_text)
    if m:
        return m.group(1) if m.group(1) else None

    # Check for standalone word
    m2 = re.search(r"(\w+)$", line_text)
    if m2:
        return m2.group(1)
    return None
