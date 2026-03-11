"""Signature help provider for TileLang code."""

from __future__ import annotations

import re
from typing import Optional

from lsprotocol import types as lsp

from .detection import is_tilelang_file
from .knowledge import lookup_symbol
from .utils import get_line_before_cursor


def build_signature_help(document, position: lsp.Position) -> Optional[lsp.SignatureHelp]:
    """Build signature help for TileLang function calls."""
    source = document.source
    if not is_tilelang_file(source):
        return None

    line_text = get_line_before_cursor(document, position)

    # Find the function call context  e.g. T.alloc_shared(
    m = re.search(r"\bT\.(\w+)\s*\(", line_text)
    if not m:
        m = re.search(r"\btilelang\.(\w+)\s*\(", line_text)
    if not m:
        return None

    sym = lookup_symbol(m.group(1))
    if not sym:
        return None

    return lsp.SignatureHelp(
        signatures=[
            lsp.SignatureInformation(
                label=sym.detail,
                documentation=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value=sym.documentation,
                ),
            )
        ],
        active_signature=0,
    )
