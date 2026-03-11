"""Tests for TileLang file detection and regex patterns."""

import pytest

from tiled_server.detection import (
    _RE_T_DOT,
    _RE_TILELANG_DOT,
    is_tilelang_file,
)


class TestFileDetection:
    def test_import_as_t(self):
        assert is_tilelang_file("import tilelang.language as T\n")

    def test_import_tilelang(self):
        assert is_tilelang_file("import tilelang\n")

    def test_from_import(self):
        assert is_tilelang_file("from tilelang.language import Kernel\n")

    def test_mixed_imports(self):
        assert is_tilelang_file("import numpy\nimport tilelang\nimport torch\n")

    def test_not_tilelang_numpy(self):
        assert not is_tilelang_file("import numpy as np\n")

    def test_not_tilelang_plain(self):
        assert not is_tilelang_file("print('hello')\n")

    def test_not_tilelang_empty(self):
        assert not is_tilelang_file("")

    def test_not_tilelang_comment_only(self):
        assert not is_tilelang_file("# import tilelang in comment\nprint(1)\n")

    def test_import_tilelang_as_alias(self):
        assert is_tilelang_file("import tilelang.language as TL\n")


class TestRegexPatterns:
    @pytest.mark.parametrize("line,expected_match", [
        ("            T.", True),
        ("    T.alloc", True),
        ("T.Kernel", True),
        ("T.", True),
        ("x = T.floa", True),
        ("print(T.", True),
        ("S.", False),
        ("no_dot_t", False),
        ("", False),
    ])
    def test_t_dot_regex(self, line, expected_match):
        m = _RE_T_DOT.search(line)
        assert bool(m) == expected_match, f"line={line!r} expected={expected_match}"

    @pytest.mark.parametrize("line,expected_match", [
        ("tilelang.", True),
        ("tilelang.jit", True),
        ("    tilelang.comp", True),
        ("x = tilelang.", True),
        ("tilelan.", False),
        ("", False),
    ])
    def test_tilelang_dot_regex(self, line, expected_match):
        m = _RE_TILELANG_DOT.search(line)
        assert bool(m) == expected_match

    def test_t_dot_captures_prefix(self):
        m = _RE_T_DOT.search("    T.alloc_sh")
        assert m.group(1) == "alloc_sh"

    def test_t_dot_empty_prefix(self):
        m = _RE_T_DOT.search("    T.")
        assert m.group(1) == ""

    def test_tilelang_dot_captures_prefix(self):
        m = _RE_TILELANG_DOT.search("    tilelang.ji")
        assert m.group(1) == "ji"
