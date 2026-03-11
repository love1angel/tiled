"""Tests for TileLang diagnostics."""

import pytest
from lsprotocol import types as lsp

from tiled_server.diagnostics import compute_diagnostics
from .conftest import GEMM_CODE, ELEMENTWISE_CODE, NON_TILELANG_CODE


class TestDiagnostics:
    def test_no_diagnostics_for_clean_gemm(self):
        diags = compute_diagnostics("file:///test.py", GEMM_CODE)
        gemm_warnings = [d for d in diags if "T.clear" in d.message]
        assert len(gemm_warnings) == 0

    def test_gemm_without_clear_warning(self):
        code = """\
import tilelang
import tilelang.language as T

@T.prim_func
def gemm(A: T.Tensor((128, 128), T.float16)):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        C_local = T.alloc_fragment((128, 128), T.float32)
        T.gemm(A, A, C_local)
"""
        diags = compute_diagnostics("file:///test.py", code)
        gemm_warnings = [d for d in diags if "T.clear" in d.message]
        assert len(gemm_warnings) == 1
        assert gemm_warnings[0].severity == lsp.DiagnosticSeverity.Information

    def test_alloc_buffer_hint(self):
        code = """\
import tilelang
import tilelang.language as T

@T.prim_func
def kernel():
    buf = T.alloc_buffer((128, 128), T.float16)
"""
        diags = compute_diagnostics("file:///test.py", code)
        hints = [d for d in diags if "T.alloc_buffer" in d.message]
        assert len(hints) == 1
        assert hints[0].severity == lsp.DiagnosticSeverity.Hint

    def test_tensor_missing_dtype_warning(self):
        code = """\
import tilelang
import tilelang.language as T

@T.prim_func
def kernel(A: T.Tensor((128, 128))):
    pass
"""
        diags = compute_diagnostics("file:///test.py", code)
        dtype_warnings = [d for d in diags if "dtype" in d.message.lower()]
        assert len(dtype_warnings) == 1
        assert dtype_warnings[0].severity == lsp.DiagnosticSeverity.Warning

    def test_no_diagnostics_for_non_tilelang(self):
        diags = compute_diagnostics("file:///test.py", NON_TILELANG_CODE)
        assert diags == []

    def test_no_diagnostics_for_correct_elementwise(self):
        diags = compute_diagnostics("file:///test.py", ELEMENTWISE_CODE)
        assert all("T.clear" not in d.message for d in diags)

    def test_diagnostic_range_is_correct(self):
        code = """\
import tilelang.language as T

@T.prim_func
def kernel():
    buf = T.alloc_buffer((4,), T.float32)
"""
        diags = compute_diagnostics("file:///test.py", code)
        hints = [d for d in diags if "alloc_buffer" in d.message]
        assert len(hints) == 1
        d = hints[0]
        assert d.range.start.line == 4
        assert d.range.start.character == code.split("\n")[4].index("T.alloc_buffer")

    def test_diagnostic_source_is_tiled(self):
        code = """\
import tilelang
@T.prim_func
def kernel():
    buf = T.alloc_buffer((4,), T.float32)
"""
        diags = compute_diagnostics("file:///test.py", code)
        for d in diags:
            assert d.source == "tiled"

    def test_multiple_gemm_no_clear(self):
        code = """\
import tilelang.language as T

@T.prim_func
def kernel():
    with T.Kernel(1, 1, threads=128) as (bx, by):
        A = T.alloc_shared((128, 128), T.float16)
        C = T.alloc_fragment((128, 128), T.float32)
        T.gemm(A, A, C)
        T.gemm(A, A, C)
"""
        diags = compute_diagnostics("file:///test.py", code)
        gemm_warnings = [d for d in diags if "T.clear" in d.message]
        assert len(gemm_warnings) >= 1

    def test_gemm_with_clear_no_warning(self):
        code = """\
import tilelang.language as T

@T.prim_func
def kernel():
    with T.Kernel(1, 1, threads=128) as (bx, by):
        A = T.alloc_shared((128, 128), T.float16)
        C = T.alloc_fragment((128, 128), T.float32)
        T.clear(C)
        T.gemm(A, A, C)
"""
        diags = compute_diagnostics("file:///test.py", code)
        gemm_warnings = [d for d in diags if "T.clear" in d.message]
        assert len(gemm_warnings) == 0

    def test_empty_source(self):
        diags = compute_diagnostics("file:///test.py", "")
        assert diags == []


class TestEdgeCases:
    def test_diagnostics_with_only_comments(self):
        code = """\
# import tilelang
# This is just a comment file
"""
        diags = compute_diagnostics("file:///test.py", code)
        assert diags == []

    def test_large_file_no_crash(self):
        lines = ["import tilelang.language as T\n"]
        lines.extend(["x = 1\n"] * 10000)
        code = "".join(lines)
        diags = compute_diagnostics("file:///test.py", code)
        assert isinstance(diags, list)

    def test_unicode_in_source(self):
        code = """\
import tilelang.language as T
# 矩阵乘法 — GEMM kernel
"""
        diags = compute_diagnostics("file:///test.py", code)
        assert isinstance(diags, list)

    def test_tab_indentation(self):
        code = "import tilelang.language as T\n\t\tT.alloc_buffer((4,), T.float32)\n"
        diags = compute_diagnostics("file:///test.py", code)
        assert isinstance(diags, list)
