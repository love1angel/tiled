"""Tests for the TileLang knowledge base."""

import pytest

from tiled_server.knowledge import (
    TileLangSymbol,
    get_t_completions,
    get_tilelang_completions,
    get_snippets,
    lookup_symbol,
)


class TestKnowledgeBase:
    def test_t_completions_count(self):
        syms = get_t_completions()
        assert len(syms) > 50

    def test_tilelang_completions_count(self):
        syms = get_tilelang_completions()
        assert len(syms) >= 3

    @pytest.mark.parametrize("name", [
        "Kernel", "alloc_shared", "alloc_fragment", "alloc_local",
        "gemm", "copy", "async_copy", "clear", "fill",
        "Pipelined", "Parallel", "Persistent", "serial", "unroll", "vectorized",
        "Tensor", "Buffer", "SharedBuffer", "FragmentBuffer",
        "ceildiv", "print", "device_assert",
        "reduce_sum", "reduce_max", "reduce_min",
        "atomic_add", "atomic_max", "atomic_min",
        "use_swizzle", "reshape", "view",
        "Layout", "Fragment", "GemmWarpPolicy",
        "dynamic", "symbolic",
    ])
    def test_core_symbol_exists(self, name):
        sym = lookup_symbol(name)
        assert sym is not None, f"Missing symbol: {name}"

    @pytest.mark.parametrize("name", [
        "Kernel", "alloc_shared", "gemm", "copy", "Pipelined", "Parallel",
    ])
    def test_core_symbol_has_documentation(self, name):
        sym = lookup_symbol(name)
        assert sym.documentation, f"Empty docs for: {name}"
        assert len(sym.documentation) > 10

    @pytest.mark.parametrize("name", [
        "Kernel", "alloc_shared", "alloc_fragment", "gemm", "Pipelined", "Parallel",
        "ceildiv", "Tensor",
    ])
    def test_core_symbol_has_snippet(self, name):
        sym = lookup_symbol(name)
        assert sym.snippet is not None, f"No snippet for: {name}"

    @pytest.mark.parametrize("dtype", [
        "float16", "float32", "float64", "bfloat16",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "half", "float", "double",
        "float8_e4m3fn", "float8_e5m2",
    ])
    def test_dtype_symbol_exists(self, dtype):
        sym = lookup_symbol(dtype)
        assert sym is not None, f"Missing dtype: {dtype}"
        assert sym.kind == "constant"

    @pytest.mark.parametrize("name", [
        "jit", "compile", "autotune", "Profiler",
    ])
    def test_tilelang_top_level_symbol(self, name):
        sym = lookup_symbol(name)
        assert sym is not None, f"Missing tilelang.{name}"

    def test_no_duplicate_symbols(self):
        syms = get_t_completions()
        names = [s.name for s in syms]
        assert len(names) == len(set(names)), f"Duplicate symbols: {[n for n in names if names.count(n) > 1]}"

    def test_symbol_kinds_valid(self):
        for sym in get_t_completions():
            assert sym.kind in ("function", "type", "constant", "decorator", "class"), \
                f"Invalid kind '{sym.kind}' for {sym.name}"

    def test_lookup_nonexistent_returns_none(self):
        assert lookup_symbol("nonexistent_xyz") is None

    def test_snippets_dict(self):
        snips = get_snippets()
        assert isinstance(snips, dict)
        assert len(snips) >= 2
        for key, val in snips.items():
            assert "prefix" in val
            assert "body" in val
            assert "description" in val
