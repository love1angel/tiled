"""Completion provider for TileLang code."""

from __future__ import annotations

from lsprotocol import types as lsp

from .detection import _RE_T_DOT, _RE_TILELANG_DOT, is_tilelang_file
from .knowledge import TileLangSymbol, get_t_completions, get_tilelang_completions
from .utils import get_line_before_cursor

# Map symbol kinds to LSP CompletionItemKind
_KIND_MAP = {
    "function": lsp.CompletionItemKind.Function,
    "type": lsp.CompletionItemKind.Class,
    "constant": lsp.CompletionItemKind.Constant,
    "decorator": lsp.CompletionItemKind.Keyword,
    "class": lsp.CompletionItemKind.Class,
}


def symbol_to_completion(sym: TileLangSymbol) -> lsp.CompletionItem:
    """Convert a TileLangSymbol to an LSP CompletionItem."""
    kind = _KIND_MAP.get(sym.kind, lsp.CompletionItemKind.Text)

    insert_text = sym.snippet if sym.snippet else sym.name
    insert_format = (
        lsp.InsertTextFormat.Snippet if sym.snippet else lsp.InsertTextFormat.PlainText
    )

    return lsp.CompletionItem(
        label=sym.name,
        kind=kind,
        detail=sym.detail,
        documentation=lsp.MarkupContent(
            kind=lsp.MarkupKind.Markdown,
            value=sym.documentation,
        ),
        insert_text=insert_text,
        insert_text_format=insert_format,
        sort_text=f"0_{sym.name}" if sym.kind != "constant" else f"1_{sym.name}",
    )


def build_completions(document, position: lsp.Position) -> lsp.CompletionList:
    """Build completion list based on cursor context."""
    source = document.source

    if not is_tilelang_file(source):
        return lsp.CompletionList(is_incomplete=False, items=[])

    line_text = get_line_before_cursor(document, position)
    items: list[lsp.CompletionItem] = []

    # T. completions
    if _RE_T_DOT.search(line_text):
        prefix = ""
        m = _RE_T_DOT.search(line_text)
        if m and m.group(1):
            prefix = m.group(1).lower()

        for sym in get_t_completions():
            if not prefix or sym.name.lower().startswith(prefix):
                items.append(symbol_to_completion(sym))

    # tilelang. completions
    elif _RE_TILELANG_DOT.search(line_text):
        prefix = ""
        m = _RE_TILELANG_DOT.search(line_text)
        if m and m.group(1):
            prefix = m.group(1).lower()

        for sym in get_tilelang_completions():
            if not prefix or sym.name.lower().startswith(prefix):
                items.append(symbol_to_completion(sym))

    # @ decorator completions
    elif line_text.rstrip().endswith("@"):
        items.append(lsp.CompletionItem(
            label="T.prim_func",
            kind=lsp.CompletionItemKind.Keyword,
            detail="@T.prim_func",
            documentation=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value="Mark a function as a TIR primitive function.",
            ),
            insert_text="T.prim_func",
        ))
        items.append(lsp.CompletionItem(
            label="tilelang.jit",
            kind=lsp.CompletionItemKind.Keyword,
            detail="@tilelang.jit(out_idx=[-1])",
            documentation=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value="JIT-compile a TileLang kernel.",
            ),
            insert_text="tilelang.jit(out_idx=[${1:-1}])",
            insert_text_format=lsp.InsertTextFormat.Snippet,
        ))

    return lsp.CompletionList(is_incomplete=False, items=items)
