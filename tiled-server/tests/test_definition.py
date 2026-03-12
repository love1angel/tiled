"""Tests for Go-to-definition provider."""

import os
import types

import pytest
from lsprotocol import types as lsp

from tiled_server.definition import build_definition

# The tilelang package root is at ~/tilelang/tilelang
TILELANG_ROOT = os.path.join(os.path.expanduser("~"), "tilelang")


def _make_workspace_folder(path: str):
    """Create a mock workspace folder with a .uri attribute."""
    return types.SimpleNamespace(uri=f"file://{path}")


def _make_document(source: str):
    """Create a mock document with .source and .lines."""
    doc = types.SimpleNamespace()
    doc.source = source
    doc.lines = source.split("\n")
    return doc


# Skip all tests if tilelang source tree is not available
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(TILELANG_ROOT, "tilelang", "language")),
    reason="tilelang source tree not found",
)

FOLDERS = [_make_workspace_folder(TILELANG_ROOT)]


# ── T.xxx (symbols in _SYMBOL_MODULE) ──────────────────────────────

class TestTDotSymbol:
    """Go-to-definition for T.alloc_shared, T.gemm, etc."""

    CODE = """\
import tilelang.language as T

x = T.alloc_shared((16, 16), T.float16)
T.gemm(A, B, C)
T.Kernel(1, 1)
T.copy(src, dst)
T.clear(buf)
"""

    def test_t_alloc_shared(self):
        doc = _make_document(self.CODE)
        # "T.alloc_shared" — cursor on 'alloc_shared'
        pos = lsp.Position(line=2, character=8)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("allocate.py")

    def test_t_gemm(self):
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=3, character=4)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("gemm_op.py")

    def test_t_kernel(self):
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=4, character=4)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("kernel.py")

    def test_t_copy(self):
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=5, character=4)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("copy_op.py")

    def test_t_clear(self):
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=6, character=4)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("fill_op.py")


# ── T.xxx.yyy (re-exported class + member via T alias) ─────────────

class TestTDotClassMember:
    """Go-to-definition for T.GemmWarpPolicy.FullRow etc."""

    CODE = """\
import tilelang.language as T

T.gemm(A, B, C, policy=T.GemmWarpPolicy.FullRow)
T.gemm(A, B, C, policy=T.GemmWarpPolicy.FullCol)
T.gemm(A, B, C, policy=T.GemmWarpPolicy.Square)
x = T.GemmWarpPolicy.FullRow
"""

    def test_t_gemm_warp_policy_class(self):
        """Cursor on 'GemmWarpPolicy' in T.GemmWarpPolicy.FullRow."""
        doc = _make_document(self.CODE)
        # line 2: "T.gemm(A, B, C, policy=T.GemmWarpPolicy.FullRow)"
        # Find "GemmWarpPolicy" start — after "policy=T."
        idx = self.CODE.split("\n")[2].index("GemmWarpPolicy")
        pos = lsp.Position(line=2, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        # Should point to the class definition line
        assert loc.range.start.line == 4  # class GemmWarpPolicy(IntEnum):

    def test_t_gemm_warp_policy_fullrow(self):
        """Cursor on 'FullRow' in T.GemmWarpPolicy.FullRow."""
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[2]
        idx = line_text.index("FullRow")
        pos = lsp.Position(line=2, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 10  # FullRow = 1

    def test_t_gemm_warp_policy_fullcol(self):
        """Cursor on 'FullCol' in T.GemmWarpPolicy.FullCol."""
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[3]
        idx = line_text.index("FullCol")
        pos = lsp.Position(line=3, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 11  # FullCol = 2

    def test_t_gemm_warp_policy_square(self):
        """Cursor on 'Square' in T.GemmWarpPolicy.Square."""
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[4]
        idx = line_text.index("Square")
        pos = lsp.Position(line=4, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 9  # Square = 0

    def test_t_gemm_warp_policy_standalone_line(self):
        """Cursor on GemmWarpPolicy in x = T.GemmWarpPolicy.FullRow."""
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[5]
        idx = line_text.index("GemmWarpPolicy")
        pos = lsp.Position(line=5, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")


# ── Imported symbol references (bare, without T.) ──────────────────

class TestImportedRef:
    """Go-to-definition for bare imported symbol (e.g. GemmWarpPolicy.FullRow)."""

    CODE = """\
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

T.gemm(A, B, C, policy=GemmWarpPolicy.FullRow)
x = GemmWarpPolicy.Square
"""

    def test_bare_gemm_warp_policy_class(self):
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[3]
        idx = line_text.index("GemmWarpPolicy")
        pos = lsp.Position(line=3, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 4

    def test_bare_gemm_warp_policy_fullrow(self):
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[3]
        idx = line_text.index("FullRow")
        pos = lsp.Position(line=3, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 10

    def test_bare_gemm_warp_policy_square(self):
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[4]
        idx = line_text.index("Square")
        pos = lsp.Position(line=4, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 9


# ── Import line resolution ──────────────────────────────────────────

class TestImportLine:
    """Go-to-definition on import statements."""

    CODE = """\
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
import tilelang
"""

    def test_import_tilelang_language_module(self):
        """Cursor on 'language' in 'import tilelang.language as T'."""
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=0, character=18)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("language/__init__.py")

    def test_from_import_module_path(self):
        """Cursor on 'tileop' in 'from tilelang.tileop.base import ...'."""
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=1, character=18)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert "tileop" in loc.uri

    def test_from_import_symbol_name(self):
        """Cursor on 'GemmWarpPolicy' in 'from ... import GemmWarpPolicy'."""
        doc = _make_document(self.CODE)
        line_text = self.CODE.split("\n")[1]
        idx = line_text.index("GemmWarpPolicy")
        pos = lsp.Position(line=1, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 4


# ── Non-tilelang files should return None ───────────────────────────

class TestNonTileLang:
    CODE = """\
import numpy as np
x = np.array([1, 2, 3])
"""

    def test_returns_none(self):
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=1, character=5)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is None


# ── tilelang.xxx (top-level symbols) ────────────────────────────────

class TestTileLangDot:
    CODE = """\
import tilelang
import tilelang.language as T

@tilelang.jit
def foo():
    pass

@tilelang.autotune(configs=[])
def bar():
    pass
"""

    def test_tilelang_jit(self):
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=3, character=12)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert "jit" in loc.uri

    def test_tilelang_autotune(self):
        doc = _make_document(self.CODE)
        pos = lsp.Position(line=7, character=12)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert "tuner" in loc.uri


# ── Custom alias (import tilelang.language as TL) ───────────────────

class TestCustomAlias:
    """Go-to-definition works with any alias, not just T."""

    CODE_TL = """\
import tilelang.language as TL

x = TL.alloc_shared((16, 16), TL.float16)
TL.gemm(A, B, C)
TL.gemm(A, B, C, policy=TL.GemmWarpPolicy.FullRow)
"""

    def test_tl_alloc_shared(self):
        doc = _make_document(self.CODE_TL)
        pos = lsp.Position(line=2, character=9)  # cursor on 'alloc_shared'
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("allocate.py")

    def test_tl_gemm(self):
        doc = _make_document(self.CODE_TL)
        pos = lsp.Position(line=3, character=5)  # cursor on 'gemm'
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("gemm_op.py")

    def test_tl_gemm_warp_policy_fullrow(self):
        doc = _make_document(self.CODE_TL)
        line_text = self.CODE_TL.split("\n")[4]
        idx = line_text.index("FullRow")
        pos = lsp.Position(line=4, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 10

    def test_tl_gemm_warp_policy_class(self):
        doc = _make_document(self.CODE_TL)
        line_text = self.CODE_TL.split("\n")[4]
        idx = line_text.index("GemmWarpPolicy")
        pos = lsp.Position(line=4, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")

    CODE_LANG = """\
import tilelang.language as lang

lang.copy(src, dst)
lang.GemmWarpPolicy.Square
"""

    def test_lang_copy(self):
        doc = _make_document(self.CODE_LANG)
        pos = lsp.Position(line=2, character=7)  # cursor on 'copy'
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("copy_op.py")

    def test_lang_gemm_warp_policy_square(self):
        doc = _make_document(self.CODE_LANG)
        line_text = self.CODE_LANG.split("\n")[3]
        idx = line_text.index("Square")
        pos = lsp.Position(line=3, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        assert loc.range.start.line == 9

    CODE_NO_ALIAS = """\
import tilelang
import numpy as np

T = something_else()
T.foo()
"""

    def test_no_alias_returns_none(self):
        """If tilelang.language is not imported, alias-based lookup should fail."""
        doc = _make_document(self.CODE_NO_ALIAS)
        pos = lsp.Position(line=4, character=3)  # cursor on 'foo'
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is None
