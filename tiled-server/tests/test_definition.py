"""Tests for Go-to-definition provider."""

import importlib.util
import os
import types

import pytest
from lsprotocol import types as lsp

from tiled_server.definition import build_definition, _find_tilelang_root


def _discover_tilelang_root() -> str | None:
    """Find tilelang package root — pip-installed or source tree."""
    # 1. pip / importlib
    try:
        spec = importlib.util.find_spec("tilelang")
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            if os.path.isdir(os.path.join(pkg_dir, "language")):
                # Return the *parent* of tilelang/ so workspace-folder mock works
                return os.path.dirname(pkg_dir)
    except (ModuleNotFoundError, ValueError):
        pass
    # 2. Common source-tree location
    src = os.path.join(os.path.expanduser("~"), "tilelang")
    if os.path.isdir(os.path.join(src, "tilelang", "language")):
        return src
    return None


TILELANG_ROOT = _discover_tilelang_root()


def _make_workspace_folder(path: str):
    """Create a mock workspace folder with a .uri attribute."""
    return types.SimpleNamespace(uri=f"file://{path}")


def _make_document(source: str):
    """Create a mock document with .source and .lines."""
    doc = types.SimpleNamespace()
    doc.source = source
    doc.lines = source.split("\n")
    return doc


def _assert_line_contains(loc: lsp.Location, expected: str):
    """Assert the resolved line in the target file contains *expected*."""
    filepath = loc.uri.replace("file://", "")
    with open(filepath) as fh:
        lines = fh.readlines()
    line_no = loc.range.start.line
    assert 0 <= line_no < len(lines), f"line {line_no} out of range"
    assert expected in lines[line_no], (
        f"expected {expected!r} at line {line_no}, got: {lines[line_no]!r}"
    )


# Skip all tests if tilelang source tree is not available
pytestmark = pytest.mark.skipif(
    TILELANG_ROOT is None,
    reason="tilelang not found (neither pip-installed nor at ~/tilelang)",
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

    def _pos(self, line_no: int, symbol: str):
        """Return a Position with character pointing inside *symbol* on *line_no*."""
        idx = self.CODE.split("\n")[line_no].index(symbol)
        return lsp.Position(line=line_no, character=idx + 1)

    def test_t_alloc_shared(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(2, "alloc_shared"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("allocate.py")

    def test_t_gemm(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(3, "gemm"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("gemm_op.py")

    def test_t_kernel(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(4, "Kernel"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("kernel.py")

    def test_t_copy(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(5, "copy"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("copy_op.py")

    def test_t_clear(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(6, "clear"), FOLDERS)
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

    def _pos(self, line_no: int, symbol: str):
        idx = self.CODE.split("\n")[line_no].index(symbol)
        return lsp.Position(line=line_no, character=idx + 2)

    def test_t_gemm_warp_policy_class(self):
        """Cursor on 'GemmWarpPolicy' in T.GemmWarpPolicy.FullRow."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(2, "GemmWarpPolicy"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "class GemmWarpPolicy")

    def test_t_gemm_warp_policy_fullrow(self):
        """Cursor on 'FullRow' in T.GemmWarpPolicy.FullRow."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(2, "FullRow"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "FullRow")

    def test_t_gemm_warp_policy_fullcol(self):
        """Cursor on 'FullCol' in T.GemmWarpPolicy.FullCol."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(3, "FullCol"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "FullCol")

    def test_t_gemm_warp_policy_square(self):
        """Cursor on 'Square' in T.GemmWarpPolicy.Square."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(4, "Square"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "Square")

    def test_t_gemm_warp_policy_standalone_line(self):
        """Cursor on GemmWarpPolicy in x = T.GemmWarpPolicy.FullRow."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(5, "GemmWarpPolicy"), FOLDERS)
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

    def _pos(self, line_no: int, symbol: str):
        idx = self.CODE.split("\n")[line_no].index(symbol)
        return lsp.Position(line=line_no, character=idx + 2)

    def test_bare_gemm_warp_policy_class(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(3, "GemmWarpPolicy"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "class GemmWarpPolicy")

    def test_bare_gemm_warp_policy_fullrow(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(3, "FullRow"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "FullRow")

    def test_bare_gemm_warp_policy_square(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(4, "Square"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "Square")


# ── Import line resolution ──────────────────────────────────────────

class TestImportLine:
    """Go-to-definition on import statements."""

    CODE = """\
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
import tilelang
"""

    def _pos(self, line_no: int, symbol: str):
        idx = self.CODE.split("\n")[line_no].index(symbol)
        return lsp.Position(line=line_no, character=idx + 2)

    def test_import_tilelang_language_module(self):
        """Cursor on 'language' in 'import tilelang.language as T'."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(0, "language"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("language/__init__.py")

    def test_import_alias(self):
        """Cursor on 'T' alias in 'import tilelang.language as T'."""
        doc = _make_document(self.CODE)
        # Find the 'T' that appears after 'as '
        line = self.CODE.split("\n")[0]
        idx = line.rindex("T")  # last T on the line (the alias)
        pos = lsp.Position(line=0, character=idx)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("language/__init__.py")

    def test_from_import_module_path(self):
        """Cursor on 'tileop' in 'from tilelang.tileop.base import ...'."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(1, "tileop"), FOLDERS)
        assert loc is not None
        assert "tileop" in loc.uri

    def test_from_import_symbol_name(self):
        """Cursor on 'GemmWarpPolicy' in 'from ... import GemmWarpPolicy'."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(1, "GemmWarpPolicy"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "class GemmWarpPolicy")


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

    def _pos(self, line_no: int, symbol: str):
        idx = self.CODE.split("\n")[line_no].index(symbol)
        return lsp.Position(line=line_no, character=idx + 1)

    def test_tilelang_jit(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(3, "jit"), FOLDERS)
        assert loc is not None
        assert "jit" in loc.uri

    def test_tilelang_autotune(self):
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(7, "autotune"), FOLDERS)
        assert loc is not None
        assert "tuner" in loc.uri


# ── tilelang.xxx.yyy (top-level class + member) ────────────────────

class TestTileLangDotClassMember:
    """Go-to-definition for tilelang.TensorSupplyType.Normal etc."""

    CODE = """\
import tilelang

profiler = fwd_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
"""

    def _pos(self, line_no: int, symbol: str):
        idx = self.CODE.split("\n")[line_no].index(symbol)
        return lsp.Position(line=line_no, character=idx + 2)

    def test_tilelang_tensor_supply_type_class(self):
        """Cursor on 'TensorSupplyType' in tilelang.TensorSupplyType.Normal."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(2, "TensorSupplyType"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("tensor.py")
        _assert_line_contains(loc, "class TensorSupplyType")

    def test_tilelang_tensor_supply_type_normal(self):
        """Cursor on 'Normal' in tilelang.TensorSupplyType.Normal."""
        doc = _make_document(self.CODE)
        loc = build_definition(doc, self._pos(2, "Normal"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("tensor.py")
        _assert_line_contains(loc, "Normal")

    CODE_SUBMODULE = """\
import tilelang

tilelang.testing.main()
"""

    def test_tilelang_submodule(self):
        """Cursor on 'testing' in tilelang.testing.main()."""
        doc = _make_document(self.CODE_SUBMODULE)
        line = self.CODE_SUBMODULE.split("\n")[2]
        idx = line.index("testing")
        pos = lsp.Position(line=2, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert "testing" in loc.uri

    def test_tilelang_submodule_func(self):
        """Cursor on 'main' in tilelang.testing.main()."""
        doc = _make_document(self.CODE_SUBMODULE)
        line = self.CODE_SUBMODULE.split("\n")[2]
        idx = line.index("main")
        pos = lsp.Position(line=2, character=idx + 2)
        loc = build_definition(doc, pos, FOLDERS)
        assert loc is not None
        assert "testing" in loc.uri
        _assert_line_contains(loc, "def main")


# ── Custom alias (import tilelang.language as TL) ───────────────────

class TestCustomAlias:
    """Go-to-definition works with any alias, not just T."""

    CODE_TL = """\
import tilelang.language as TL

x = TL.alloc_shared((16, 16), TL.float16)
TL.gemm(A, B, C)
TL.gemm(A, B, C, policy=TL.GemmWarpPolicy.FullRow)
"""

    def _pos(self, code: str, line_no: int, symbol: str):
        idx = code.split("\n")[line_no].index(symbol)
        return lsp.Position(line=line_no, character=idx + 2)

    def test_tl_alloc_shared(self):
        doc = _make_document(self.CODE_TL)
        loc = build_definition(doc, self._pos(self.CODE_TL, 2, "alloc_shared"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("allocate.py")

    def test_tl_gemm(self):
        doc = _make_document(self.CODE_TL)
        loc = build_definition(doc, self._pos(self.CODE_TL, 3, "gemm"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("gemm_op.py")

    def test_tl_gemm_warp_policy_fullrow(self):
        doc = _make_document(self.CODE_TL)
        loc = build_definition(doc, self._pos(self.CODE_TL, 4, "FullRow"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "FullRow")

    def test_tl_gemm_warp_policy_class(self):
        doc = _make_document(self.CODE_TL)
        loc = build_definition(doc, self._pos(self.CODE_TL, 4, "GemmWarpPolicy"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")

    CODE_LANG = """\
import tilelang.language as lang

lang.copy(src, dst)
lang.GemmWarpPolicy.Square
"""

    def test_lang_copy(self):
        doc = _make_document(self.CODE_LANG)
        loc = build_definition(doc, self._pos(self.CODE_LANG, 2, "copy"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("copy_op.py")

    def test_lang_gemm_warp_policy_square(self):
        doc = _make_document(self.CODE_LANG)
        loc = build_definition(doc, self._pos(self.CODE_LANG, 3, "Square"), FOLDERS)
        assert loc is not None
        assert loc.uri.endswith("base.py")
        _assert_line_contains(loc, "Square")

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


# ── _find_tilelang_root discovery ───────────────────────────────────

class TestFindTileLangRoot:
    """_find_tilelang_root should work with workspace folders and pip."""

    def test_workspace_folder(self):
        """Should find tilelang when workspace IS the tilelang repo."""
        folders = [_make_workspace_folder(TILELANG_ROOT)]
        root = _find_tilelang_root(folders)
        assert root is not None
        assert os.path.isdir(os.path.join(root, "language"))

    def test_child_workspace(self):
        """Should find tilelang when workspace is a subdirectory."""
        child = os.path.join(TILELANG_ROOT, "examples")
        if not os.path.isdir(child):
            pytest.skip("examples/ directory not present")
        folders = [_make_workspace_folder(child)]
        root = _find_tilelang_root(folders)
        assert root is not None
        assert os.path.isdir(os.path.join(root, "language"))

    def test_unrelated_workspace_still_finds(self):
        """With unrelated workspace, should still find via pip or source."""
        folders = [_make_workspace_folder("/tmp")]
        root = _find_tilelang_root(folders)
        # This will find it via importlib if pip-installed,
        # otherwise None (source-tree only works via workspace match).
        # We don't assert not-None because pip install may not exist.
        if root is not None:
            assert os.path.isdir(os.path.join(root, "language"))

    def test_empty_folders(self):
        """With no workspace folders, should still find via pip if installed."""
        root = _find_tilelang_root([])
        if root is not None:
            assert os.path.isdir(os.path.join(root, "language"))

    def test_definition_works_with_empty_folders(self):
        """Go-to-definition should work even without workspace folders
        if tilelang is pip-installed."""
        root = _find_tilelang_root([])
        if root is None:
            pytest.skip("tilelang not pip-installed, cannot test empty folders")
        doc = _make_document(
            "import tilelang.language as T\n\nT.gemm(A, B, C)\n"
        )
        pos = lsp.Position(line=2, character=4)
        loc = build_definition(doc, pos, [])
        assert loc is not None
        assert loc.uri.endswith("gemm_op.py")
