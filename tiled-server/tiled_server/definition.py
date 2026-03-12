"""Go-to-definition provider for TileLang code."""

from __future__ import annotations

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
    # print_op.py
    "print": "print_op.py",
    "device_assert": "print_op.py",
    # customize.py
    "use_swizzle": "customize.py",
    "annotate_layout": "customize.py",
    "use_swizzle_panel": "customize.py",
    # symbolics.py
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
    """Find the tilelang package root in workspace or nearby directories."""
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

    # Try T.xxx pattern
    for m in re.finditer(r"\bT\.(\w+)", line):
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

    # Try tilelang.xxx pattern
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

    return None
