"""TileLang file detection — regex patterns and helpers."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns for detecting tilelang usage
# ---------------------------------------------------------------------------
_RE_IMPORT_T = re.compile(
    r"""^[^#]*(?:from\s+tilelang\.language\s+import|import\s+tilelang\.language\s+as\s+\w+)""",
    re.MULTILINE,
)
_RE_IMPORT_TILELANG = re.compile(r"""^[^#]*import\s+tilelang""", re.MULTILINE)

# Trigger patterns for completions
_RE_T_DOT = re.compile(r"""\bT\.(\w*)$""")
_RE_TILELANG_DOT = re.compile(r"""\btilelang\.(\w*)$""")

_RE_PRIM_FUNC_NO_KERNEL = re.compile(
    r"""@T\.prim_func\s*\ndef\s+\w+\(.*?\).*?:(?:(?!T\.Kernel).)*$""",
    re.DOTALL,
)


def is_tilelang_file(source: str) -> bool:
    """Check if a Python file imports tilelang."""
    return bool(_RE_IMPORT_T.search(source) or _RE_IMPORT_TILELANG.search(source))
