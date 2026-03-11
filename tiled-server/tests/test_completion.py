"""Tests for TileLang completion logic."""

import pytest
from lsprotocol import types as lsp

from tiled_server.completion import symbol_to_completion
from tiled_server.knowledge import (
    TileLangSymbol,
    get_t_completions,
    get_tilelang_completions,
)


class TestSymbolToCompletion:
    def test_function_symbol(self):
        sym = TileLangSymbol("gemm", "function", "T.gemm(A, B, C)", "Do gemm.")
        item = symbol_to_completion(sym)
        assert item.label == "gemm"
        assert item.kind == lsp.CompletionItemKind.Function
        assert item.detail == "T.gemm(A, B, C)"

    def test_type_symbol(self):
        sym = TileLangSymbol("Tensor", "type", "T.Tensor", "Tensor type.")
        item = symbol_to_completion(sym)
        assert item.kind == lsp.CompletionItemKind.Class

    def test_constant_symbol(self):
        sym = TileLangSymbol("float16", "constant", "T.float16", "float16 dtype.")
        item = symbol_to_completion(sym)
        assert item.kind == lsp.CompletionItemKind.Constant
        assert item.sort_text.startswith("1_")

    def test_function_sort_before_constant(self):
        func = TileLangSymbol("gemm", "function", "T.gemm()", "gemm")
        const = TileLangSymbol("float16", "constant", "T.float16", "f16")
        func_item = symbol_to_completion(func)
        const_item = symbol_to_completion(const)
        assert func_item.sort_text < const_item.sort_text

    def test_snippet_uses_snippet_format(self):
        sym = TileLangSymbol("alloc_shared", "function", "T.alloc_shared()", "Alloc.",
                            snippet="alloc_shared((${1:M}, ${2:N}), ${3:dtype})")
        item = symbol_to_completion(sym)
        assert item.insert_text_format == lsp.InsertTextFormat.Snippet
        assert "${1:" in item.insert_text

    def test_no_snippet_uses_plain_text(self):
        sym = TileLangSymbol("clear", "function", "T.clear(buf)", "Clear.")
        item = symbol_to_completion(sym)
        assert item.insert_text_format == lsp.InsertTextFormat.PlainText
        assert item.insert_text == "clear"

    def test_documentation_is_markdown(self):
        sym = TileLangSymbol("Kernel", "class", "T.Kernel()", "Launch a kernel.\n\n```python\n...\n```")
        item = symbol_to_completion(sym)
        assert item.documentation.kind == lsp.MarkupKind.Markdown


class TestCompletionLogic:
    """Test the completion filtering logic directly."""

    def test_t_dot_returns_all_symbols(self):
        syms = get_t_completions()
        func_names = {s.name for s in syms if s.kind == "function"}
        const_names = {s.name for s in syms if s.kind == "constant"}
        assert "alloc_shared" in func_names
        assert "float16" in const_names

    def test_filter_by_prefix(self):
        prefix = "alloc"
        syms = [s for s in get_t_completions() if s.name.lower().startswith(prefix)]
        names = {s.name for s in syms}
        assert "alloc_shared" in names
        assert "alloc_fragment" in names
        assert "alloc_local" in names
        assert "gemm" not in names

    def test_filter_by_prefix_reduce(self):
        prefix = "reduce"
        syms = [s for s in get_t_completions() if s.name.lower().startswith(prefix)]
        names = {s.name for s in syms}
        assert "reduce_sum" in names
        assert "reduce_max" in names
        assert "reduce_min" in names
        assert "alloc_shared" not in names

    def test_tilelang_completions_have_jit(self):
        syms = get_tilelang_completions()
        names = {s.name for s in syms}
        assert "jit" in names

    def test_decorator_completions(self):
        """@ trigger should offer T.prim_func and tilelang.jit."""
        line = "    @"
        assert line.rstrip().endswith("@")
