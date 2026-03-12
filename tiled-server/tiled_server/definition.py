"""Go-to-definition provider for TileLang code."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Optional

from lsprotocol import types as lsp

from .detection import is_tilelang_file

logger = logging.getLogger(__name__)

# Map T.<symbol> to relative module path inside tilelang/language/
_SYMBOL_MODULE: dict[str, str] = {
    # allocate.py
    "alloc_shared": "allocate.py",
    "alloc_fragment": "allocate.py",
    "alloc_local": "allocate.py",
    "alloc_var": "allocate.py",
    "alloc_barrier": "allocate.py",
    "alloc_cluster_barrier": "allocate.py",
    "alloc_tmem": "allocate.py",
    "alloc_reducer": "allocate.py",
    "alloc_descriptor": "allocate.py",
    "alloc_wgmma_desc": "allocate.py",
    "alloc_tcgen05_smem_desc": "allocate.py",
    "alloc_tcgen05_instr_desc": "allocate.py",
    "empty": "allocate.py",
    # copy_op.py
    "copy": "copy_op.py",
    "async_copy": "copy_op.py",
    "c2d_im2col": "copy_op.py",
    # gemm_op.py
    "gemm": "gemm_op.py",
    "gemm_v1": "gemm_op.py",
    "gemm_v2": "gemm_op.py",
    # fill_op.py
    "fill": "fill_op.py",
    "clear": "fill_op.py",
    # kernel.py
    "Kernel": "kernel.py",
    "KernelLaunchFrame": "kernel.py",
    "get_thread_binding": "kernel.py",
    "get_thread_bindings": "kernel.py",
    "get_block_binding": "kernel.py",
    "get_block_bindings": "kernel.py",
    # loop.py
    "Parallel": "loop.py",
    "Pipelined": "loop.py",
    "Persistent": "loop.py",
    "serial": "loop.py",
    "Serial": "loop.py",
    "unroll": "loop.py",
    "Unroll": "loop.py",
    "vectorized": "loop.py",
    "Vectorized": "loop.py",
    # proxy.py
    "Tensor": "proxy.py",
    "Buffer": "proxy.py",
    "SharedBuffer": "proxy.py",
    "LocalBuffer": "proxy.py",
    "FragmentBuffer": "proxy.py",
    "StridedTensor": "proxy.py",
    "make_tensor": "proxy.py",
    "ptr": "proxy.py",
    # reduce_op.py
    "reduce": "reduce_op.py",
    "reduce_sum": "reduce_op.py",
    "reduce_max": "reduce_op.py",
    "reduce_min": "reduce_op.py",
    "reduce_abssum": "reduce_op.py",
    "reduce_absmax": "reduce_op.py",
    "reduce_bitand": "reduce_op.py",
    "reduce_bitor": "reduce_op.py",
    "reduce_bitxor": "reduce_op.py",
    "cumsum": "reduce_op.py",
    "finalize_reducer": "reduce_op.py",
    "warp_reduce_sum": "reduce_op.py",
    "warp_reduce_max": "reduce_op.py",
    "warp_reduce_min": "reduce_op.py",
    "warp_reduce_bitand": "reduce_op.py",
    "warp_reduce_bitor": "reduce_op.py",
    # print_op.py
    "print": "print_op.py",
    "device_assert": "print_op.py",
    # annotations.py
    "use_swizzle": "annotations.py",
    "annotate_layout": "annotations.py",
    "annotate_safe_value": "annotations.py",
    "annotate_l2_hit_ratio": "annotations.py",
    "annotate_restrict_buffers": "annotations.py",
    # customize.py
    "dp4a": "customize.py",
    "clamp": "customize.py",
    "reshape": "customize.py",
    "view": "customize.py",
    "loop_break": "customize.py",
    # atomic.py
    "atomic_add": "atomic.py",
    "atomic_addx2": "atomic.py",
    "atomic_addx4": "atomic.py",
    "atomic_max": "atomic.py",
    "atomic_min": "atomic.py",
    "atomic_load": "atomic.py",
    "atomic_store": "atomic.py",
    # logical.py
    "any_of": "logical.py",
    "all_of": "logical.py",
    # warpgroup.py
    "ws": "warpgroup.py",
    # utils.py
    "index_to_coordinates": "utils.py",
    # random.py
    "rng_init": "random.py",
    "rng_rand": "random.py",
    "rng_rand_float": "random.py",
    # pdl.py
    "pdl_trigger": "pdl.py",
    "pdl_sync": "pdl.py",
    # cluster.py
    "cluster_arrive_relaxed": "cluster.py",
    "cluster_arrive": "cluster.py",
    "cluster_wait": "cluster.py",
    "cluster_sync": "cluster.py",
    "cluster_rank_in_cluster": "cluster.py",
    "dynamic": "symbolics.py",
    "symbolic": "symbolics.py",
    # frame.py
    "has_let_value": "frame.py",
    "get_let_value": "frame.py",
    # tir/entry.py
    "prim_func": "tir/entry.py",
    # tir/op.py
    "if_then_else": "tir/op.py",
    "infinity": "tir/op.py",
    "ceildiv": "tir/op.py",
    "floordiv": "tir/op.py",
    "exp": "tir/op.py",
    "log": "tir/op.py",
    "max": "tir/op.py",
    "min": "tir/op.py",
    # atomic.py
    "atomic_add": "atomic.py",
    "atomic_addx2": "atomic.py",
    # experimental
    "gemm_sp": "experimental/gemm_sp.py",
    "gemm_sp_v2": "experimental/gemm_sp.py",
}

# Map tilelang.<symbol> to (subpath, file) relative to tilelang package root
_TOPLEVEL_MODULE: dict[str, str] = {
    "autotune": "autotuner/tuner.py",
    "jit": "jit/__init__.py",
    "TensorSupplyType": "utils/tensor.py",
    "prim_func": "language/tir/entry.py",
}


def _find_tilelang_root(workspace_folders: list) -> Optional[str]:
    """Find the tilelang package root.

    Search order:
    1. Workspace folders (developer working on tilelang source)
    2. pip-installed package via importlib
    """
    # 1. Workspace folders
    for folder in workspace_folders:
        uri = folder.uri if hasattr(folder, "uri") else str(folder)
        path = uri.replace("file://", "")
        # Check if this workspace IS tilelang
        candidate = os.path.join(path, "tilelang", "language")
        if os.path.isdir(candidate):
            return os.path.join(path, "tilelang")
        # Check parent dir (e.g. workspace is ~/tilelang/examples)
        parent = os.path.dirname(path)
        candidate2 = os.path.join(parent, "tilelang", "language")
        if os.path.isdir(candidate2):
            return os.path.join(parent, "tilelang")

    # 2. pip-installed package
    try:
        spec = importlib.util.find_spec("tilelang")
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            if os.path.isdir(os.path.join(pkg_dir, "language")):
                return pkg_dir
    except (ModuleNotFoundError, ValueError):
        pass

    return None


def _find_def_in_file(filepath: str, symbol_name: str) -> Optional[int]:
    """Find the line number where a symbol is defined in a file."""
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        patterns = [
            re.compile(rf"^def {re.escape(symbol_name)}\b"),
            re.compile(rf"^class {re.escape(symbol_name)}\b"),
            re.compile(rf"^{re.escape(symbol_name)}\s*="),
            re.compile(rf"^\s+def {re.escape(symbol_name)}\b"),
            re.compile(rf"^\s+class {re.escape(symbol_name)}\b"),
        ]
        for i, line in enumerate(lines):
            for pat in patterns:
                if pat.search(line):
                    return i
    except OSError:
        pass
    return None


def _find_language_alias(lines: list[str]) -> Optional[str]:
    """Find the alias used for ``import tilelang.language as <alias>``.

    Returns the alias string (e.g. ``"T"``, ``"TL"``) or *None* if the
    import is not present.
    """
    for line in lines:
        m = re.match(r"^\s*import\s+tilelang\.language\s+as\s+(\w+)", line)
        if m:
            return m.group(1)
    return None


def build_definition(
    document, position: lsp.Position, workspace_folders: list
) -> Optional[lsp.Location]:
    """Resolve go-to-definition for T.xxx symbols."""
    source = document.source
    if not is_tilelang_file(source):
        return None

    lines = document.lines if hasattr(document, "lines") else source.split("\n")
    if position.line >= len(lines):
        return None

    line = lines[position.line]

    # Detect the alias used for tilelang.language (e.g. T, TL, lang, ...)
    alias = _find_language_alias(lines)

    # Try alias.xxx.yyy pattern (e.g. T.GemmWarpPolicy.FullRow)
    if alias:
        pat_cls = re.compile(rf"\b{re.escape(alias)}\.(\w+)\.(\w+)")
        for m in pat_cls.finditer(line):
            full_start = m.start(1)
            full_end = m.end(2)
            if full_start <= position.character <= full_end:
                class_name = m.group(1)
                member_name = m.group(2)
                on_member = position.character >= m.start(2)
                root = _find_tilelang_root(workspace_folders)
                if root:
                    loc = _resolve_t_reexport(root, class_name)
                    if loc and on_member:
                        filepath = loc.uri.replace("file://", "")
                        member_line = _find_member_in_file(filepath, member_name)
                        if member_line is not None:
                            return lsp.Location(
                                uri=loc.uri,
                                range=lsp.Range(
                                    start=lsp.Position(line=member_line, character=0),
                                    end=lsp.Position(line=member_line, character=0),
                                ),
                            )
                    if loc:
                        return loc
                break

    # Try alias.xxx pattern (e.g. T.gemm, TL.alloc_shared)
    if alias:
        pat_sym = re.compile(rf"\b{re.escape(alias)}\.(\w+)")
        for m in pat_sym.finditer(line):
            sym_start, sym_end = m.start(1), m.end(1)
            if sym_start <= position.character <= sym_end:
                symbol_name = m.group(1)
                module_file = _SYMBOL_MODULE.get(symbol_name)
                if module_file:
                    root = _find_tilelang_root(workspace_folders)
                    if root:
                        filepath = os.path.join(root, "language", module_file)
                        line_no = _find_def_in_file(filepath, symbol_name)
                        if line_no is not None:
                            uri = f"file://{filepath}"
                            return lsp.Location(
                                uri=uri,
                                range=lsp.Range(
                                    start=lsp.Position(line=line_no, character=0),
                                    end=lsp.Position(line=line_no, character=0),
                                ),
                            )
                break

    # Try tilelang.xxx.yyy pattern (e.g. tilelang.TensorSupplyType.Normal)
    if not line.lstrip().startswith(("from ", "import ")):
        for m in re.finditer(r"\btilelang\.(\w+)\.(\w+)", line):
            full_start = m.start(1)
            full_end = m.end(2)
            if full_start <= position.character <= full_end:
                class_name = m.group(1)
                member_name = m.group(2)
                on_member = position.character >= m.start(2)
                module_file = _TOPLEVEL_MODULE.get(class_name)
                root = _find_tilelang_root(workspace_folders)
                if root and module_file:
                    filepath = os.path.join(root, module_file)
                    class_line = _find_def_in_file(filepath, class_name)
                    if class_line is not None:
                        uri = f"file://{filepath}"
                        if on_member:
                            member_line = _find_member_in_file(filepath, member_name)
                            if member_line is not None:
                                return lsp.Location(
                                    uri=uri,
                                    range=lsp.Range(
                                        start=lsp.Position(line=member_line, character=0),
                                        end=lsp.Position(line=member_line, character=0),
                                    ),
                                )
                        return lsp.Location(
                            uri=uri,
                            range=lsp.Range(
                                start=lsp.Position(line=class_line, character=0),
                                end=lsp.Position(line=class_line, character=0),
                            ),
                        )
                break

    # Try tilelang.xxx pattern (in non-import expressions like @tilelang.jit)
    if not line.lstrip().startswith(("from ", "import ")):
        for m in re.finditer(r"\btilelang\.(\w+)", line):
            sym_start, sym_end = m.start(1), m.end(1)
            if sym_start <= position.character <= sym_end:
                symbol_name = m.group(1)
                module_file = _TOPLEVEL_MODULE.get(symbol_name)
                if module_file:
                    root = _find_tilelang_root(workspace_folders)
                    if root:
                        filepath = os.path.join(root, module_file)
                        line_no = _find_def_in_file(filepath, symbol_name)
                        if line_no is not None:
                            uri = f"file://{filepath}"
                            return lsp.Location(
                                uri=uri,
                                range=lsp.Range(
                                    start=lsp.Position(line=line_no, character=0),
                                    end=lsp.Position(line=line_no, character=0),
                                ),
                            )
                break

    # Handle import statements: from/import tilelang.xxx.yyy [as Z | import Name]
    root = _find_tilelang_root(workspace_folders)
    if root:
        loc = _resolve_import_line(root, line, position.character)
        if loc:
            return loc

    # Resolve bare references to imported symbols (e.g. GemmWarpPolicy.FullRow)
    if root:
        loc = _resolve_imported_ref(root, lines, line, position.character)
        if loc:
            return loc

    return None


def _resolve_module_path(root: str, module_path: str) -> Optional[lsp.Location]:
    """Resolve a dotted module path to its __init__.py or .py file."""
    parts = module_path.split(".")
    rel_parts = parts[1:]  # drop "tilelang"
    if not rel_parts:
        # Just "tilelang" itself
        filepath = os.path.join(root, "__init__.py")
        if os.path.isfile(filepath):
            return lsp.Location(
                uri=f"file://{filepath}",
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=0),
                ),
            )
        return None

    base = os.path.join(root, *rel_parts)
    # Package directory
    init = os.path.join(base, "__init__.py")
    if os.path.isfile(init):
        return lsp.Location(
            uri=f"file://{init}",
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=0),
            ),
        )
    # Module file
    mod_file = base + ".py"
    if os.path.isfile(mod_file):
        return lsp.Location(
            uri=f"file://{mod_file}",
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=0),
            ),
        )
    return None


def _resolve_import_line(
    root: str, line: str, character: int
) -> Optional[lsp.Location]:
    """Handle all import-line definition lookups."""
    # "from tilelang.xxx.yyy import Name1, Name2"
    m_from = re.match(r"^from\s+(tilelang(?:\.\w+)*)\s+import\s+(.+)", line)
    if m_from:
        module_path = m_from.group(1)
        imports_str = m_from.group(2)

        # Check if cursor is on the module path
        mod_start = m_from.start(1)
        mod_end = m_from.end(1)
        if mod_start <= character <= mod_end:
            # Find which dotted segment the cursor is on
            prefix = _dotted_prefix_at(module_path, character - mod_start)
            return _resolve_module_path(root, prefix)

        # Check if cursor is on an imported name
        for m_name in re.finditer(r"\b(\w+)\b", imports_str):
            abs_start = m_from.start(2) + m_name.start(1)
            abs_end = m_from.start(2) + m_name.end(1)
            if abs_start <= character <= abs_end:
                symbol_name = m_name.group(1)
                return _resolve_import(root, module_path, symbol_name)
        return None

    # "import tilelang.xxx.yyy [as Z]"
    m_imp = re.match(r"^import\s+(tilelang(?:\.\w+)*)(?:\s+as\s+\w+)?", line)
    if m_imp:
        module_path = m_imp.group(1)
        mod_start = m_imp.start(1)
        mod_end = m_imp.end(1)
        if mod_start <= character <= mod_end:
            prefix = _dotted_prefix_at(module_path, character - mod_start)
            return _resolve_module_path(root, prefix)

    return None


def _dotted_prefix_at(dotted: str, offset: int) -> str:
    """Given 'tilelang.foo.bar' and an offset, return the prefix up to that segment."""
    # Walk segments to find which one the offset falls in
    pos = 0
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        seg_end = pos + len(part)
        if offset <= seg_end:
            return ".".join(parts[: i + 1])
        pos = seg_end + 1  # skip the dot
    return dotted


def _resolve_import(
    root: str, module_path: str, symbol_name: str
) -> Optional[lsp.Location]:
    """Resolve 'from tilelang.x.y import Name' to a source Location."""
    # Convert tilelang.x.y -> x/y relative to tilelang root
    parts = module_path.split(".")
    rel_parts = parts[1:]  # drop "tilelang"
    base = os.path.join(root, *rel_parts) if rel_parts else root

    # Try: base is a package dir, symbol in __init__.py or a submodule
    candidates = []
    if os.path.isdir(base):
        candidates.append(os.path.join(base, "__init__.py"))
        # symbol might be a submodule file (e.g. tilelang.tools has Analyzer.py)
        candidates.append(os.path.join(base, f"{symbol_name}.py"))
        # lowercase variant
        candidates.append(os.path.join(base, f"{symbol_name.lower()}.py"))
    # Try: base.py is a module file
    candidates.append(base + ".py")

    for filepath in candidates:
        line_no = _find_def_in_file(filepath, symbol_name)
        if line_no is not None:
            uri = f"file://{filepath}"
            return lsp.Location(
                uri=uri,
                range=lsp.Range(
                    start=lsp.Position(line=line_no, character=0),
                    end=lsp.Position(line=line_no, character=0),
                ),
            )
    return None


def _resolve_t_reexport(
    root: str, symbol_name: str
) -> Optional[lsp.Location]:
    """Resolve a symbol accessed via T.xxx by scanning language/__init__.py re-exports."""
    init_path = os.path.join(root, "language", "__init__.py")
    if not os.path.isfile(init_path):
        return None
    try:
        with open(init_path, "r", encoding="utf-8") as f:
            init_lines = f.readlines()
    except OSError:
        return None

    for src_line in init_lines:
        m = re.match(r"^from\s+([\w.]+)\s+import\s+(.+)", src_line)
        if not m:
            continue
        for n in re.finditer(r"\b(\w+)\b", m.group(2)):
            if n.group(1) == symbol_name:
                module_path = m.group(1)
                # Resolve relative imports (leading dot)
                if not module_path.startswith("tilelang"):
                    # relative import like ".tileop.base" — not a dotted module path
                    # Reconstruct as absolute: tilelang.language.<rel>
                    continue
                return _resolve_import(root, module_path, symbol_name)
    return None


def _resolve_imported_ref(
    root: str, all_lines: list[str], line: str, character: int
) -> Optional[lsp.Location]:
    """Resolve a bare imported symbol reference like GemmWarpPolicy.FullRow."""
    # Match Identifier or Identifier.member at cursor
    for m in re.finditer(r"\b(\w+)(?:\.(\w+))?\b", line):
        name_start, name_end = m.start(1), m.end(1)
        member = m.group(2)
        full_end = m.end(0)
        if not (name_start <= character <= full_end):
            continue
        symbol_name = m.group(1)
        on_member = member and character > m.start(2) - 1

        # Scan file imports for this symbol
        module_path = None
        for src_line in all_lines:
            # from tilelang.xxx import ..., Symbol, ...
            im = re.match(
                r"^from\s+(tilelang(?:\.\w+)*)\s+import\s+(.+)", src_line
            )
            if im:
                for n in re.finditer(r"\b(\w+)\b", im.group(2)):
                    if n.group(1) == symbol_name:
                        module_path = im.group(1)
                        break
            if module_path:
                break

        if not module_path:
            continue

        # Resolve the class/function location
        loc = _resolve_import(root, module_path, symbol_name)
        if not loc:
            continue

        if on_member and member:
            # Find member inside the resolved file
            filepath = loc.uri.replace("file://", "")
            member_line = _find_member_in_file(filepath, member)
            if member_line is not None:
                return lsp.Location(
                    uri=loc.uri,
                    range=lsp.Range(
                        start=lsp.Position(line=member_line, character=0),
                        end=lsp.Position(line=member_line, character=0),
                    ),
                )
        return loc
    return None


def _find_member_in_file(filepath: str, member_name: str) -> Optional[int]:
    """Find a class member (attribute, method, enum value) in a file."""
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        patterns = [
            re.compile(rf"^\s+{re.escape(member_name)}\s*="),
            re.compile(rf"^\s+def {re.escape(member_name)}\b"),
            re.compile(rf"^\s+class {re.escape(member_name)}\b"),
        ]
        for i, line in enumerate(lines):
            for pat in patterns:
                if pat.search(line):
                    return i
    except OSError:
        pass
    return None
