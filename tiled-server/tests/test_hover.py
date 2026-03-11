"""Tests for TileLang hover and signature help logic."""

import pytest

from tiled_server.knowledge import lookup_symbol


class TestHoverLogic:
    """Test the hover lookup logic directly."""

    def test_hover_t_symbol(self):
        sym = lookup_symbol("alloc_shared")
        assert sym is not None
        assert "shared memory" in sym.documentation.lower()

    def test_hover_gemm(self):
        sym = lookup_symbol("gemm")
        assert sym is not None
        assert "matrix" in sym.documentation.lower() or "gemm" in sym.documentation.lower()

    def test_hover_pipelined(self):
        sym = lookup_symbol("Pipelined")
        assert sym is not None
        assert "pipeline" in sym.documentation.lower()

    def test_hover_kernel(self):
        sym = lookup_symbol("Kernel")
        assert sym is not None
        assert "kernel" in sym.documentation.lower() or "GPU" in sym.documentation

    def test_hover_tilelang_jit(self):
        sym = lookup_symbol("jit")
        assert sym is not None
        assert "JIT" in sym.documentation or "jit" in sym.documentation.lower()


class TestSignatureLogic:
    """Test that symbols have proper detail strings usable for signature help."""

    @pytest.mark.parametrize("name,expected_fragment", [
        ("alloc_shared", "shape"),
        ("Kernel", "threads"),
        ("Pipelined", "num_stages"),
        ("Tensor", "shape"),
        ("gemm", "A"),
        ("copy", "src"),
    ])
    def test_detail_contains_params(self, name, expected_fragment):
        sym = lookup_symbol(name)
        assert sym is not None
        assert expected_fragment in sym.detail
