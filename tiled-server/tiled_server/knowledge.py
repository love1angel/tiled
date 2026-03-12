"""TileLang DSL knowledge base — completions, docs, signatures, templates, and tips.

Unified knowledge base for the LSP server (completions, hover, signatures)
and the MCP server (API index, templates, optimization tips, best practices).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TileLangSymbol:
    name: str
    kind: str  # "function", "type", "constant", "decorator", "class"
    detail: str  # short signature or type info
    documentation: str  # markdown docs
    snippet: Optional[str] = None  # VS Code snippet body
    category: Optional[str] = None
    deprecated: bool = False


# ---------------------------------------------------------------------------
# Categories for grouping symbols (used by MCP API index)
# ---------------------------------------------------------------------------
CATEGORIES: dict[str, str] = {
    "memory": "Memory allocation functions (alloc_shared, alloc_fragment, etc.)",
    "data_movement": "Data copy operations (copy, async_copy)",
    "compute": "Compute operations (gemm, fill, clear)",
    "loop": "Loop constructs (Kernel, Pipelined, Parallel, serial, unroll)",
    "reduction": "Reduction operations (reduce_sum, reduce_max, warp_reduce_*)",
    "atomic": "Atomic operations (atomic_add, atomic_max, atomic_min)",
    "utility": "Utility functions (ceildiv, print, reshape, clamp)",
    "types": "Buffer and tensor types (Tensor, Buffer, Layout, Fragment)",
    "annotations": "Memory annotations (use_swizzle, annotate_layout)",
    "threading": "Thread/block binding queries",
    "cluster": "Cluster synchronization primitives",
    "random": "Random number generation",
    "dtypes": "Scalar data types (float16, float32, int8, etc.)",
    "tilelang": "Top-level tilelang.* symbols (jit, autotune, compile)",
}

# ---------------------------------------------------------------------------
# Data types accessible via T.<dtype>
# ---------------------------------------------------------------------------
_SCALAR_DTYPES = [
    "bool", "int4", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "bfloat16",
    "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
    "float8_e8m0fnu", "float4_e2m1fn",
    # aliases
    "half", "float", "double", "short", "int", "uint", "long",
]

# ---------------------------------------------------------------------------
# Core API symbols  (T.xxx)
# ---------------------------------------------------------------------------
_SYMBOLS: list[TileLangSymbol] = [
    # -- Decorators / top-level --
    TileLangSymbol(
        "prim_func", "decorator", "@T.prim_func",
        "Decorator that marks a function as a TIR primitive function.\n\n"
        "Used inside a `@tilelang.jit` wrapper to define GPU kernels.",
        category="types",
    ),
    TileLangSymbol(
        "macro", "decorator", "@T.macro",
        "Decorator that defines a reusable kernel macro.",
        category="types",
    ),

    # -- Kernel launch --
    TileLangSymbol(
        "Kernel", "class",
        "T.Kernel(*grid_dims, threads=128)",
        "Create a GPU kernel launch context.\n\n"
        "```python\n"
        "with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):\n"
        "    ...\n"
        "```\n\n"
        "**Parameters**\n"
        "- `*grid_dims`: Grid dimensions (block indices).\n"
        "- `threads`: Number of threads per block (default 128).",
        snippet="Kernel(${1:grid_x}, ${2:grid_y}, threads=${3:128}) as (${4:bx}, ${5:by}):",
        category="loop",
    ),

    # -- Tensor type --
    TileLangSymbol(
        "Tensor", "type",
        "T.Tensor(shape, dtype)",
        "Type annotation for kernel tensor parameters.\n\n"
        "```python\n"
        "def kernel(A: T.Tensor((M, N), T.float16)):\n"
        "```",
        snippet='Tensor((${1:M}, ${2:N}), ${3:T.float16})',
        category="types",
    ),

    # -- Buffer types --
    TileLangSymbol(
        "Buffer", "type", "T.Buffer",
        "General buffer type for kernel parameters.",
        category="types",
    ),
    TileLangSymbol(
        "SharedBuffer", "type", "T.SharedBuffer",
        "Buffer in shared memory scope.",
        category="types",
    ),
    TileLangSymbol(
        "LocalBuffer", "type", "T.LocalBuffer",
        "Buffer in local (thread-private) memory scope.",
        category="types",
    ),
    TileLangSymbol(
        "FragmentBuffer", "type", "T.FragmentBuffer",
        "Buffer for tensor core fragment storage.",
        category="types",
    ),

    # -- Memory allocation --
    TileLangSymbol(
        "alloc_shared", "function",
        "T.alloc_shared(shape, dtype) -> Buffer",
        "Allocate a shared memory buffer for inter-thread communication.\n\n"
        "```python\n"
        "A_shared = T.alloc_shared((block_M, block_K), T.float16)\n"
        "```\n\n"
        "**Parameters**\n"
        "- `shape`: Tuple of dimensions\n"
        "- `dtype`: Data type (e.g. T.float16)\n\n"
        "Shared memory is visible to all threads within a block and is "
        "much faster than global memory.",
        snippet="alloc_shared((${1:block_M}, ${2:block_K}), ${3:dtype})",
        category="memory",
    ),
    TileLangSymbol(
        "alloc_fragment", "function",
        "T.alloc_fragment(shape, dtype) -> Buffer",
        "Allocate a fragment buffer for register-level / tensor core storage.\n\n"
        "```python\n"
        "C_local = T.alloc_fragment((block_M, block_N), T.float32)\n"
        "```\n\n"
        "**Parameters**\n"
        "- `shape`: Tuple of dimensions\n"
        "- `dtype`: Data type\n\n"
        "Fragments are stored in registers and used for accumulation "
        "in GEMM and other tensor core operations.",
        snippet="alloc_fragment((${1:block_M}, ${2:block_N}), ${3:dtype})",
        category="memory",
    ),
    TileLangSymbol(
        "alloc_local", "function",
        "T.alloc_local(shape, dtype) -> Buffer",
        "Allocate thread-private local memory.\n\n"
        "```python\n"
        "tmp = T.alloc_local((4,), T.float32)\n"
        "```",
        snippet="alloc_local((${1:shape},), ${2:dtype})",
        category="memory",
    ),
    TileLangSymbol(
        "alloc_var", "function",
        "T.alloc_var(dtype) -> Buffer",
        "Allocate a single-element variable buffer.",
        snippet="alloc_var(${1:dtype})",
        category="memory",
    ),
    TileLangSymbol(
        "alloc_barrier", "function",
        "T.alloc_barrier() -> Buffer",
        "Allocate a barrier for synchronization.",
        category="memory",
    ),
    TileLangSymbol(
        "alloc_reducer", "function",
        "T.alloc_reducer(dtype) -> Buffer",
        "Allocate a reducer for reduction operations.",
        category="memory",
    ),
    TileLangSymbol(
        "alloc_tmem", "function",
        "T.alloc_tmem(shape, dtype) -> Buffer",
        "Allocate tensor memory (SM100+).",
        snippet="alloc_tmem((${1:shape},), ${2:dtype})",
        category="memory",
    ),
    TileLangSymbol(
        "empty", "function",
        "T.empty(shape, dtype) -> Buffer",
        "Allocate an uninitialized buffer.",
        snippet="empty((${1:shape},), ${2:dtype})",
        category="memory",
    ),

    # -- Data movement --
    TileLangSymbol(
        "copy", "function",
        "T.copy(src, dst, coalesced_width=None)",
        "Copy data between buffers or between global/shared memory.\n\n"
        "```python\n"
        "T.copy(A[by * block_M, k * block_K], A_shared)\n"
        "T.copy(C_local, C[by * block_M, bx * block_N])\n"
        "```\n\n"
        "**Parameters**\n"
        "- `src`: Source buffer or slice\n"
        "- `dst`: Destination buffer or slice\n"
        "- `coalesced_width`: Memory coalescing width hint\n"
        "- `disable_tma`: Disable TMA acceleration\n"
        "- `eviction_policy`: Cache eviction policy",
        snippet="copy(${1:src}, ${2:dst})",
        category="data_movement",
    ),
    TileLangSymbol(
        "async_copy", "function",
        "T.async_copy(src, dst)",
        "Asynchronous copy (cp.async) between global and shared memory.\n\n"
        "Requires explicit synchronization with barriers.",
        snippet="async_copy(${1:src}, ${2:dst})",
        category="data_movement",
    ),

    # -- Compute operations --
    TileLangSymbol(
        "gemm", "function",
        "T.gemm(A, B, C, transpose_A=False, transpose_B=False, policy=None)",
        "Perform matrix multiplication C += A @ B using tensor cores.\n\n"
        "```python\n"
        "T.gemm(A_shared, B_shared, C_local)\n"
        "T.gemm(A_shared, B_shared, C_local, transpose_B=True, "
        "policy=T.GemmWarpPolicy.FullRow)\n"
        "```\n\n"
        "**Parameters**\n"
        "- `A`: Left operand (shared memory)\n"
        "- `B`: Right operand (shared memory)\n"
        "- `C`: Accumulator (fragment)\n"
        "- `transpose_A`: Transpose A before multiplication\n"
        "- `transpose_B`: Transpose B before multiplication\n"
        "- `policy`: Warp partition policy (GemmWarpPolicy.FullRow, FullCol, Square)\n"
        "- `k_pack`: Packed cores for ROCm\n"
        "- `clear_accum`: Clear accumulator before GEMM",
        snippet="gemm(${1:A_shared}, ${2:B_shared}, ${3:C_local})",
        category="compute",
    ),
    TileLangSymbol(
        "gemm_v1", "function",
        "T.gemm_v1(A, B, C, ...)",
        "Alternative GEMM implementation (v1 backend).",
        category="compute",
    ),
    TileLangSymbol(
        "gemm_v2", "function",
        "T.gemm_v2(A, B, C, ...)",
        "Alternative GEMM implementation (v2 backend).",
        category="compute",
    ),
    TileLangSymbol(
        "gemm_sp", "function",
        "T.gemm_sp(A, B, C, ...)",
        "Sparse GEMM with 2:4 structured sparsity support.",
        category="compute",
    ),
    TileLangSymbol(
        "fill", "function",
        "T.fill(buffer, value)",
        "Fill a buffer with a constant value.",
        snippet="fill(${1:buffer}, ${2:0})",
        category="compute",
    ),
    TileLangSymbol(
        "clear", "function",
        "T.clear(buffer)",
        "Clear (zero-initialize) a buffer. Commonly used before GEMM accumulation.\n\n"
        "```python\n"
        "T.clear(C_local)\n"
        "```",
        snippet="clear(${1:buffer})",
        category="compute",
    ),

    # -- Loop constructs --
    TileLangSymbol(
        "Pipelined", "function",
        "T.Pipelined(extent, num_stages=N)",
        "Create a software-pipelined loop for overlapping compute and memory.\n\n"
        "```python\n"
        "for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):\n"
        "    T.copy(A[..., k * block_K], A_shared)\n"
        "    T.gemm(A_shared, B_shared, C_local)\n"
        "```\n\n"
        "**Parameters**\n"
        "- `extent` or `(start, stop)`: Iteration range\n"
        "- `num_stages`: Number of pipeline stages (typically 2-4)",
        snippet="Pipelined(${1:extent}, num_stages=${2:3})",
        category="loop",
    ),
    TileLangSymbol(
        "Parallel", "function",
        "T.Parallel(*extents)",
        "Create nested parallel loops for element-wise operations.\n\n"
        "```python\n"
        "for i, j in T.Parallel(block_M, block_N):\n"
        "    C[i, j] = A[i, j] + B[i, j]\n"
        "```\n\n"
        "All loop indices are bound in parallel across threads.",
        snippet="Parallel(${1:block_M}, ${2:block_N})",
        category="loop",
    ),
    TileLangSymbol(
        "Persistent", "function",
        "T.Persistent(*extents)",
        "Create a persistent loop that keeps a thread block alive across iterations.",
        snippet="Persistent(${1:extent})",
        category="loop",
    ),
    TileLangSymbol(
        "serial", "function",
        "T.serial(start, stop)",
        "Create a serial (sequential) for loop.",
        snippet="serial(${1:0}, ${2:N})",
        category="loop",
    ),
    TileLangSymbol(
        "unroll", "function",
        "T.unroll(start, stop)",
        "Create a fully unrolled for loop (compile-time unrolling).",
        snippet="unroll(${1:0}, ${2:N})",
        category="loop",
    ),
    TileLangSymbol(
        "vectorized", "function",
        "T.vectorized(start, stop)",
        "Create a vectorized for loop.",
        snippet="vectorized(${1:0}, ${2:N})",
        category="loop",
    ),

    # -- Reduction --
    TileLangSymbol(
        "reduce_sum", "function",
        "T.reduce_sum(src, dst, dim=None, clear=True)",
        "Sum reduction along a dimension.",
        snippet="reduce_sum(${1:src}, ${2:dst}, dim=${3:0})",
        category="reduction",
    ),
    TileLangSymbol(
        "reduce_max", "function",
        "T.reduce_max(src, dst, dim=None, clear=True)",
        "Max reduction along a dimension.",
        snippet="reduce_max(${1:src}, ${2:dst}, dim=${3:0})",
        category="reduction",
    ),
    TileLangSymbol(
        "reduce_min", "function",
        "T.reduce_min(src, dst, dim=None, clear=True)",
        "Min reduction along a dimension.",
        snippet="reduce_min(${1:src}, ${2:dst}, dim=${3:0})",
        category="reduction",
    ),
    TileLangSymbol(
        "reduce_abssum", "function",
        "T.reduce_abssum(src, dst, dim=None)",
        "Absolute-value sum reduction.",
        category="reduction",
    ),
    TileLangSymbol(
        "reduce_absmax", "function",
        "T.reduce_absmax(src, dst, dim=None)",
        "Absolute-value max reduction.",
        category="reduction",
    ),
    TileLangSymbol(
        "reduce", "function",
        "T.reduce(op, src, dst, dim=None)",
        "Generic reduction with a given binary operator.",
        category="reduction",
    ),
    TileLangSymbol(
        "cumsum", "function",
        "T.cumsum(src, dst, dim=None)",
        "Compute cumulative sum along a dimension.",
        category="reduction",
    ),
    TileLangSymbol(
        "finalize_reducer", "function",
        "T.finalize_reducer(reducer)",
        "Finalize a reducer after reduction iterations.",
        category="reduction",
    ),
    TileLangSymbol(
        "warp_reduce_sum", "function",
        "T.warp_reduce_sum(value)",
        "Warp-level sum reduction.",
        category="reduction",
    ),
    TileLangSymbol(
        "warp_reduce_max", "function",
        "T.warp_reduce_max(value)",
        "Warp-level max reduction.",
        category="reduction",
    ),
    TileLangSymbol(
        "warp_reduce_min", "function",
        "T.warp_reduce_min(value)",
        "Warp-level min reduction.",
        category="reduction",
    ),

    # -- Atomic operations --
    TileLangSymbol(
        "atomic_add", "function",
        "T.atomic_add(dst, value)",
        "Atomically add a value to a memory location.",
        snippet="atomic_add(${1:dst}, ${2:value})",
        category="atomic",
    ),
    TileLangSymbol(
        "atomic_max", "function",
        "T.atomic_max(dst, value)",
        "Atomically compute max.",
        category="atomic",
    ),
    TileLangSymbol(
        "atomic_min", "function",
        "T.atomic_min(dst, value)",
        "Atomically compute min.",
        category="atomic",
    ),

    # -- Utility --
    TileLangSymbol(
        "ceildiv", "function",
        "T.ceildiv(a, b) -> int",
        "Compute ceiling division: `(a + b - 1) // b`.",
        snippet="ceildiv(${1:a}, ${2:b})",
        category="utility",
    ),
    TileLangSymbol(
        "print", "function",
        "T.print(fmt, *args)",
        "Device-side printf for debugging.\n\n"
        "```python\n"
        'T.print(\"val = {}\", val)\n'
        "```",
        category="utility",
    ),
    TileLangSymbol(
        "device_assert", "function",
        "T.device_assert(cond, msg)",
        "Assert a condition on the device side.",
        category="utility",
    ),
    TileLangSymbol(
        "if_then_else", "function",
        "T.if_then_else(cond, true_val, false_val)",
        "Conditional expression (ternary).\n\n"
        "```python\n"
        "result = T.if_then_else(x > 0, x, 0)\n"
        "```",
        category="utility",
    ),
    TileLangSymbol(
        "exp", "function",
        "T.exp(x)",
        "Compute exponential function.",
        category="utility",
    ),
    TileLangSymbol(
        "log", "function",
        "T.log(x)",
        "Compute natural logarithm.",
        category="utility",
    ),
    TileLangSymbol(
        "max", "function",
        "T.max(a, b)",
        "Compute element-wise maximum.",
        category="utility",
    ),
    TileLangSymbol(
        "min", "function",
        "T.min(a, b)",
        "Compute element-wise minimum.",
        category="utility",
    ),
    TileLangSymbol(
        "abs", "function",
        "T.abs(x)",
        "Compute absolute value.",
        category="utility",
    ),
    TileLangSymbol(
        "infinity", "function",
        "T.infinity(dtype)",
        "Return positive infinity for the given dtype.",
        category="utility",
    ),
    TileLangSymbol(
        "floordiv", "function",
        "T.floordiv(a, b)",
        "Floor division.",
        category="utility",
    ),

    # -- Memory annotations --
    TileLangSymbol(
        "use_swizzle", "function",
        "T.use_swizzle(panel_size, enable=True)",
        "Enable shared memory swizzle pattern for bank conflict avoidance.\n\n"
        "```python\n"
        "T.use_swizzle(panel_size=10, enable=True)\n"
        "```",
        snippet="use_swizzle(${1:nbit})",
        category="annotations",
    ),
    TileLangSymbol(
        "annotate_layout", "function",
        "T.annotate_layout(buffer, layout)",
        "Annotate a buffer with a custom layout.",
        category="annotations",
    ),

    # -- Thread/block queries --
    TileLangSymbol(
        "get_thread_binding", "function",
        "T.get_thread_binding() -> int",
        "Get the current thread index within the block (threadIdx.x).",
        category="threading",
    ),
    TileLangSymbol(
        "get_thread_bindings", "function",
        "T.get_thread_bindings() -> tuple",
        "Get all thread binding indices.",
        category="threading",
    ),
    TileLangSymbol(
        "get_block_binding", "function",
        "T.get_block_binding() -> int",
        "Get the current block index.",
        category="threading",
    ),
    TileLangSymbol(
        "get_block_bindings", "function",
        "T.get_block_bindings() -> tuple",
        "Get all block binding indices.",
        category="threading",
    ),

    # -- Misc --
    TileLangSymbol(
        "reshape", "function",
        "T.reshape(buffer, new_shape)",
        "Reshape a buffer to a new shape.",
        snippet="reshape(${1:buffer}, (${2:new_shape},))",
        category="utility",
    ),
    TileLangSymbol(
        "view", "function",
        "T.view(buffer, new_shape)",
        "View a buffer with a new shape (no copy).",
        category="utility",
    ),
    TileLangSymbol(
        "clamp", "function",
        "T.clamp(value, min_val, max_val)",
        "Clamp a value between min and max.",
        category="utility",
    ),
    TileLangSymbol(
        "dp4a", "function",
        "T.dp4a(a, b, c)",
        "4-element dot product with accumulate (int8).",
        category="utility",
    ),
    TileLangSymbol(
        "loop_break", "function",
        "T.loop_break()",
        "Break out of the current loop.",
        category="utility",
    ),
    TileLangSymbol(
        "any_of", "function",
        "T.any_of(*args)",
        "Logical OR across arguments.",
        category="utility",
    ),
    TileLangSymbol(
        "all_of", "function",
        "T.all_of(*args)",
        "Logical AND across arguments.",
        category="utility",
    ),
    TileLangSymbol(
        "dynamic", "function",
        "T.dynamic(name)",
        "Declare a dynamic (runtime) symbolic variable.",
        category="utility",
    ),
    TileLangSymbol(
        "symbolic", "function",
        "T.symbolic(name)",
        "Declare a compile-time symbolic variable.",
        category="utility",
    ),
    TileLangSymbol(
        "import_source", "function",
        "T.import_source(source)",
        "Import raw C source code into the kernel.",
        category="utility",
    ),
    TileLangSymbol(
        "make_tensor", "function",
        "T.make_tensor(ptr, shape, dtype)",
        "Create a tensor from a raw pointer.",
        category="utility",
    ),
    TileLangSymbol(
        "ptr", "function",
        "T.ptr(dtype)",
        "Create a pointer type annotation.",
        category="utility",
    ),
    TileLangSymbol(
        "index_to_coordinates", "function",
        "T.index_to_coordinates(index, shape)",
        "Convert a flat index to multi-dimensional coordinates.",
        category="utility",
    ),

    # -- Random --
    TileLangSymbol(
        "rng_init", "function",
        "T.rng_init(seed)",
        "Initialize the random number generator state.",
        category="random",
    ),
    TileLangSymbol(
        "rng_rand", "function",
        "T.rng_rand(state)",
        "Generate a random integer.",
        category="random",
    ),
    TileLangSymbol(
        "rng_rand_float", "function",
        "T.rng_rand_float(state)",
        "Generate a random float in [0, 1).",
        category="random",
    ),

    # -- Cluster --
    TileLangSymbol(
        "cluster_sync", "function",
        "T.cluster_sync()",
        "Synchronize all blocks in a cluster.",
        category="cluster",
    ),
    TileLangSymbol(
        "cluster_arrive", "function",
        "T.cluster_arrive()",
        "Signal cluster arrival.",
        category="cluster",
    ),
    TileLangSymbol(
        "cluster_wait", "function",
        "T.cluster_wait()",
        "Wait for cluster arrival.",
        category="cluster",
    ),
    TileLangSymbol(
        "block_rank_in_cluster", "function",
        "T.block_rank_in_cluster()",
        "Get the rank of the current block within its cluster.",
        category="cluster",
    ),

    # -- PDL --
    TileLangSymbol(
        "pdl_trigger", "function",
        "T.pdl_trigger()",
        "Trigger a PDL (Programmatic Dependent Launch) event.",
        category="utility",
    ),
    TileLangSymbol(
        "pdl_sync", "function",
        "T.pdl_sync()",
        "Synchronize with PDL events.",
        category="utility",
    ),

    # -- Warp specialization --
    TileLangSymbol(
        "ws", "class",
        "T.ws",
        "Warp specialization utilities. Access via `T.ws.*`.",
        category="types",
    ),

    # -- Layout --
    TileLangSymbol(
        "Layout", "class",
        "T.Layout(mapping, shape)",
        "Define a data layout for buffer or loop mapping.",
        category="types",
    ),
    TileLangSymbol(
        "Fragment", "class",
        "T.Fragment(mapping, shape)",
        "Define a fragment layout for tensor core operations.",
        category="types",
    ),

    # -- Load/store intrinsics --
    TileLangSymbol("__ldg", "function", "T.__ldg(ptr)", "Load through texture cache (read-only).", category="utility"),
    TileLangSymbol("ldg32", "function", "T.ldg32(ptr)", "32-bit load through texture cache.", category="utility"),
    TileLangSymbol("ldg64", "function", "T.ldg64(ptr)", "64-bit load through texture cache.", category="utility"),
    TileLangSymbol("ldg128", "function", "T.ldg128(ptr)", "128-bit load through texture cache.", category="utility"),
    TileLangSymbol("ldg256", "function", "T.ldg256(ptr)", "256-bit load through texture cache.", category="utility"),
    TileLangSymbol("stg32", "function", "T.stg32(ptr, val)", "32-bit global store.", category="utility"),
    TileLangSymbol("stg64", "function", "T.stg64(ptr, val)", "64-bit global store.", category="utility"),
    TileLangSymbol("stg128", "function", "T.stg128(ptr, val)", "128-bit global store.", category="utility"),
    TileLangSymbol("stg256", "function", "T.stg256(ptr, val)", "256-bit global store.", category="utility"),

    # -- GemmWarpPolicy --
    TileLangSymbol(
        "GemmWarpPolicy", "class",
        "T.GemmWarpPolicy",
        "Warp partition policy for GEMM operations.\n\n"
        "**Members**\n"
        "- `GemmWarpPolicy.FullRow` — Each warp processes a full row\n"
        "- `GemmWarpPolicy.FullCol` — Each warp processes a full column\n"
        "- `GemmWarpPolicy.Square` — Square warp tiling",
        category="types",
    ),

    # -- Kernel launch frame --
    TileLangSymbol(
        "KernelLaunchFrame", "class",
        "T.KernelLaunchFrame",
        "Low-level kernel launch frame (advanced usage).",
        category="types",
    ),
]

# Build dtype symbols
for _dt in _SCALAR_DTYPES:
    _SYMBOLS.append(TileLangSymbol(
        _dt, "constant", f"T.{_dt}",
        f"Data type: `{_dt}`",
        category="dtypes",
    ))

# ---------------------------------------------------------------------------
# Top-level tilelang symbols (tilelang.xxx)
# ---------------------------------------------------------------------------
_TILELANG_SYMBOLS: list[TileLangSymbol] = [
    TileLangSymbol(
        "jit", "decorator",
        "@tilelang.jit(out_idx=[-1], ...)",
        "JIT-compile a TileLang kernel function.\n\n"
        "```python\n"
        "@tilelang.jit(out_idx=[-1])\n"
        "def my_kernel(M, N, K, ...):\n"
        "    @T.prim_func\n"
        "    def impl(A: T.Tensor(...), ...):\n"
        "        ...\n"
        "    return impl\n"
        "```\n\n"
        "**Parameters**\n"
        "- `out_idx`: Indices of output tensors (negative = from end).\n"
        "- `target`: Target device ('cuda', 'hip', etc.)",
        snippet="jit(out_idx=[${1:-1}])",
        category="tilelang",
    ),
    TileLangSymbol(
        "compile", "function",
        "tilelang.compile(func, ...)",
        "Compile a TileLang function without JIT.",
        category="tilelang",
    ),
    TileLangSymbol(
        "autotune", "decorator",
        "@tilelang.autotune(configs=[], ...)",
        "Auto-tune a kernel over a set of configurations.\n\n"
        "```python\n"
        "@tilelang.autotune(configs=get_configs(), cache_input_tensors=True)\n"
        "@tilelang.jit(out_idx=[-1])\n"
        "def my_kernel(M, N, K, block_M: int, block_N: int, ...):\n"
        "    ...\n"
        "```",
        category="tilelang",
    ),
    TileLangSymbol(
        "Profiler", "class",
        "tilelang.Profiler",
        "Profiler for benchmarking compiled kernels.",
        category="tilelang",
    ),
    TileLangSymbol(
        "TensorSupplyType", "class",
        "tilelang.TensorSupplyType",
        "Tensor supply type for profiler.\n\n"
        "**Members**\n"
        "- `TensorSupplyType.Normal` — Normal random tensors\n"
        "- `TensorSupplyType.Integer` — Integer tensors\n"
        "- `TensorSupplyType.Randn` — Standard normal distribution",
        category="tilelang",
    ),
]

# ---------------------------------------------------------------------------
# Build indexes (for MCP API lookups by name / category)
# ---------------------------------------------------------------------------
_ALL_SYMBOLS = _SYMBOLS + _TILELANG_SYMBOLS
_BY_NAME: dict[str, TileLangSymbol] = {s.name: s for s in _ALL_SYMBOLS}
_BY_CATEGORY: dict[str, list[TileLangSymbol]] = {}
for _s in _ALL_SYMBOLS:
    if _s.category:
        _BY_CATEGORY.setdefault(_s.category, []).append(_s)

# ---------------------------------------------------------------------------
# Snippet templates for common patterns (used by LSP)
# ---------------------------------------------------------------------------
_SNIPPETS = {
    "tilelang_gemm_kernel": {
        "prefix": "tl-gemm",
        "description": "TileLang GEMM kernel template",
        "body": (
            'import tilelang\n'
            'import tilelang.language as T\n'
            '\n'
            '\n'
            '@tilelang.jit(out_idx=[-1])\n'
            'def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):\n'
            '    @T.prim_func\n'
            '    def gemm(\n'
            '        A: T.Tensor((M, K), dtype),\n'
            '        B: T.Tensor((K, N), dtype),\n'
            '        C: T.Tensor((M, N), dtype),\n'
            '    ):\n'
            '        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):\n'
            '            A_shared = T.alloc_shared((block_M, block_K), dtype)\n'
            '            B_shared = T.alloc_shared((block_K, block_N), dtype)\n'
            '            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)\n'
            '\n'
            '            T.clear(C_local)\n'
            '            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):\n'
            '                T.copy(A[by * block_M, k * block_K], A_shared)\n'
            '                T.copy(B[k * block_K, bx * block_N], B_shared)\n'
            '                T.gemm(A_shared, B_shared, C_local)\n'
            '\n'
            '            T.copy(C_local, C[by * block_M, bx * block_N])\n'
            '\n'
            '    return gemm\n'
        ),
    },
    "tilelang_elementwise_kernel": {
        "prefix": "tl-elementwise",
        "description": "TileLang element-wise kernel template",
        "body": (
            'import tilelang\n'
            'import tilelang.language as T\n'
            '\n'
            '\n'
            '@tilelang.jit(out_idx=[-1])\n'
            'def elementwise_op(M, N, block_M, block_N, dtype=T.float32):\n'
            '    @T.prim_func\n'
            '    def kernel(\n'
            '        A: T.Tensor((M, N), dtype),\n'
            '        B: T.Tensor((M, N), dtype),\n'
            '        C: T.Tensor((M, N), dtype),\n'
            '    ):\n'
            '        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):\n'
            '            A_shared = T.alloc_shared((block_M, block_N), dtype)\n'
            '            B_shared = T.alloc_shared((block_M, block_N), dtype)\n'
            '            C_local = T.alloc_fragment((block_M, block_N), dtype)\n'
            '\n'
            '            T.copy(A[by * block_M, bx * block_N], A_shared)\n'
            '            T.copy(B[by * block_M, bx * block_N], B_shared)\n'
            '            for i, j in T.Parallel(block_M, block_N):\n'
            '                C_local[i, j] = A_shared[i, j] + B_shared[i, j]\n'
            '            T.copy(C_local, C[by * block_M, bx * block_N])\n'
            '\n'
            '    return kernel\n'
        ),
    },
    "tilelang_kernel_skeleton": {
        "prefix": "tl-kernel",
        "description": "TileLang kernel skeleton",
        "body": (
            '@T.prim_func\n'
            'def ${1:kernel_name}(\n'
            '    ${2:A}: T.Tensor((${3:M}, ${4:N}), ${5:dtype}),\n'
            '):\n'
            '    with T.Kernel(${6:grid_x}, ${7:grid_y}, threads=${8:128}) as (bx, by):\n'
            '        ${0:pass}\n'
        ),
    },
}

# ---------------------------------------------------------------------------
# Kernel templates (full code examples, used by MCP and extension)
# ---------------------------------------------------------------------------
TEMPLATES: dict[str, dict] = {
    "gemm": {
        "name": "GEMM (Matrix Multiplication)",
        "description": "Dense matrix multiplication kernel with pipelined shared memory loading.",
        "code": (
            "import tilelang\n"
            "import tilelang.language as T\n"
            "\n"
            "\n"
            "@tilelang.jit(out_idx=[-1])\n"
            "def matmul(M, N, K, block_M=128, block_N=128, block_K=32,\n"
            "           dtype=T.float16, accum_dtype=T.float32):\n"
            "    @T.prim_func\n"
            "    def gemm(\n"
            "        A: T.Tensor((M, K), dtype),\n"
            "        B: T.Tensor((K, N), dtype),\n"
            "        C: T.Tensor((M, N), dtype),\n"
            "    ):\n"
            "        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):\n"
            "            A_shared = T.alloc_shared((block_M, block_K), dtype)\n"
            "            B_shared = T.alloc_shared((block_K, block_N), dtype)\n"
            "            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)\n"
            "\n"
            "            T.clear(C_local)\n"
            "            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):\n"
            "                T.copy(A[by * block_M, k * block_K], A_shared)\n"
            "                T.copy(B[k * block_K, bx * block_N], B_shared)\n"
            "                T.gemm(A_shared, B_shared, C_local)\n"
            "\n"
            "            T.copy(C_local, C[by * block_M, bx * block_N])\n"
            "\n"
            "    return gemm\n"
        ),
    },
    "elementwise": {
        "name": "Element-wise Operation",
        "description": "Element-wise kernel template (add, mul, etc.).",
        "code": (
            "import tilelang\n"
            "import tilelang.language as T\n"
            "\n"
            "\n"
            "@tilelang.jit(out_idx=[-1])\n"
            "def elementwise_add(M, N, block_M=64, block_N=64, dtype=T.float32):\n"
            "    @T.prim_func\n"
            "    def kernel(\n"
            "        A: T.Tensor((M, N), dtype),\n"
            "        B: T.Tensor((M, N), dtype),\n"
            "        C: T.Tensor((M, N), dtype),\n"
            "    ):\n"
            "        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):\n"
            "            A_shared = T.alloc_shared((block_M, block_N), dtype)\n"
            "            B_shared = T.alloc_shared((block_M, block_N), dtype)\n"
            "            C_local = T.alloc_fragment((block_M, block_N), dtype)\n"
            "\n"
            "            T.copy(A[by * block_M, bx * block_N], A_shared)\n"
            "            T.copy(B[by * block_M, bx * block_N], B_shared)\n"
            "            for i, j in T.Parallel(block_M, block_N):\n"
            "                C_local[i, j] = A_shared[i, j] + B_shared[i, j]\n"
            "            T.copy(C_local, C[by * block_M, bx * block_N])\n"
            "\n"
            "    return kernel\n"
        ),
    },
    "reduction": {
        "name": "Reduction Kernel",
        "description": "Row-wise reduction (e.g. sum, max) kernel template.",
        "code": (
            "import tilelang\n"
            "import tilelang.language as T\n"
            "\n"
            "\n"
            "@tilelang.jit(out_idx=[-1])\n"
            "def row_sum(M, N, block_M=64, block_N=128, dtype=T.float32):\n"
            "    @T.prim_func\n"
            "    def kernel(\n"
            "        A: T.Tensor((M, N), dtype),\n"
            "        Out: T.Tensor((M,), dtype),\n"
            "    ):\n"
            "        with T.Kernel(T.ceildiv(M, block_M), threads=128) as (bx,):\n"
            "            A_shared = T.alloc_shared((block_M, block_N), dtype)\n"
            "            acc = T.alloc_fragment((block_M,), dtype)\n"
            "            row_sum = T.alloc_fragment((block_M,), dtype)\n"
            "\n"
            "            T.clear(acc)\n"
            "            for k in T.serial(0, T.ceildiv(N, block_N)):\n"
            "                T.copy(A[bx * block_M, k * block_N], A_shared)\n"
            "                T.reduce_sum(A_shared, row_sum, dim=1)\n"
            "                for i in T.Parallel(block_M):\n"
            "                    acc[i] += row_sum[i]\n"
            "\n"
            "            T.copy(acc, Out[bx * block_M])\n"
            "\n"
            "    return kernel\n"
        ),
    },
    "softmax": {
        "name": "Online Softmax",
        "description": "Numerically stable online softmax kernel.",
        "code": (
            "import tilelang\n"
            "import tilelang.language as T\n"
            "\n"
            "\n"
            "@tilelang.jit(out_idx=[-1])\n"
            "def softmax(M, N, block_N=128, dtype=T.float16, accum_dtype=T.float32):\n"
            "    @T.prim_func\n"
            "    def kernel(\n"
            "        A: T.Tensor((M, N), dtype),\n"
            "        Out: T.Tensor((M, N), dtype),\n"
            "    ):\n"
            "        with T.Kernel(M, threads=128) as (bx,):\n"
            "            A_shared = T.alloc_shared((block_N,), dtype)\n"
            "            acc = T.alloc_fragment((block_N,), accum_dtype)\n"
            "            m_val = T.alloc_fragment((1,), accum_dtype)\n"
            "            l_val = T.alloc_fragment((1,), accum_dtype)\n"
            "            row_max = T.alloc_fragment((1,), accum_dtype)\n"
            "\n"
            "            T.fill(m_val, -T.infinity(accum_dtype))\n"
            "            T.fill(l_val, 0)\n"
            "\n"
            "            # Pass 1: find max\n"
            "            for k in T.serial(0, T.ceildiv(N, block_N)):\n"
            "                T.copy(A[bx, k * block_N], A_shared)\n"
            "                T.reduce_max(A_shared, row_max, dim=0)\n"
            "                for i in T.Parallel(1):\n"
            "                    m_val[i] = T.max(m_val[i], row_max[i])\n"
            "\n"
            "            # Pass 2: compute exp and sum\n"
            "            for k in T.serial(0, T.ceildiv(N, block_N)):\n"
            "                T.copy(A[bx, k * block_N], A_shared)\n"
            "                for i in T.Parallel(block_N):\n"
            "                    acc[i] = T.exp(A_shared[i] - m_val[0])\n"
            "                    T.atomic_add(l_val[0], acc[i])\n"
            "\n"
            "            # Pass 3: normalize\n"
            "            for k in T.serial(0, T.ceildiv(N, block_N)):\n"
            "                T.copy(A[bx, k * block_N], A_shared)\n"
            "                for i in T.Parallel(block_N):\n"
            "                    Out[bx, k * block_N + i] = T.exp(A_shared[i] - m_val[0]) / l_val[0]\n"
            "\n"
            "    return kernel\n"
        ),
    },
    "flash_attention": {
        "name": "Flash Attention (Forward)",
        "description": "Flash Attention v2 forward pass kernel template.",
        "code": (
            "import tilelang\n"
            "import tilelang.language as T\n"
            "\n"
            "\n"
            "@tilelang.jit(out_idx=[-1])\n"
            "def flash_attention_fwd(batch, heads, seq_len, dim,\n"
            "                        block_M=64, block_N=64, num_stages=2,\n"
            "                        dtype=T.float16, accum_dtype=T.float32):\n"
            "    scale = (1.0 / dim) ** 0.5\n"
            "\n"
            "    @T.prim_func\n"
            "    def kernel(\n"
            "        Q: T.Tensor((batch, seq_len, heads, dim), dtype),\n"
            "        K: T.Tensor((batch, seq_len, heads, dim), dtype),\n"
            "        V: T.Tensor((batch, seq_len, heads, dim), dtype),\n"
            "        O: T.Tensor((batch, seq_len, heads, dim), dtype),\n"
            "    ):\n"
            "        with T.Kernel(T.ceildiv(seq_len, block_M), batch * heads, threads=128) as (bx, byz):\n"
            "            bz = byz // heads\n"
            "            by = byz % heads\n"
            "\n"
            "            Q_shared = T.alloc_shared((block_M, dim), dtype)\n"
            "            K_shared = T.alloc_shared((block_N, dim), dtype)\n"
            "            V_shared = T.alloc_shared((block_N, dim), dtype)\n"
            "            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)\n"
            "            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)\n"
            "            m_i = T.alloc_fragment((block_M,), accum_dtype)\n"
            "            l_i = T.alloc_fragment((block_M,), accum_dtype)\n"
            "\n"
            "            T.clear(acc_o)\n"
            "            T.fill(m_i, -T.infinity(accum_dtype))\n"
            "            T.fill(l_i, 0)\n"
            "\n"
            "            T.copy(Q[bz, bx * block_M, by, :], Q_shared)\n"
            "\n"
            "            for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=num_stages):\n"
            "                T.copy(K[bz, k * block_N, by, :], K_shared)\n"
            "                T.copy(V[bz, k * block_N, by, :], V_shared)\n"
            "\n"
            "                T.clear(acc_s)\n"
            "                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)\n"
            "\n"
            "                # Scale and online softmax update\n"
            "                for i, j in T.Parallel(block_M, block_N):\n"
            "                    acc_s[i, j] *= scale\n"
            "\n"
            "                # ... (full implementation requires online softmax tracking)\n"
            "                # See examples/flash_attention/ for complete implementation\n"
            "\n"
            "            T.copy(acc_o, O[bz, bx * block_M, by, :])\n"
            "\n"
            "    return kernel\n"
        ),
    },
    "autotune_gemm": {
        "name": "Autotuned GEMM",
        "description": "Matrix multiplication with autotune over block sizes and thread counts.",
        "code": (
            "import tilelang\n"
            "import tilelang.language as T\n"
            "import itertools\n"
            "import torch\n"
            "\n"
            "\n"
            "def get_configs():\n"
            '    """Generate autotune search space."""\n'
            "    iter_params = dict(\n"
            "        block_M=[64, 128, 256],\n"
            "        block_N=[64, 128, 256],\n"
            "        block_K=[32, 64],\n"
            "        num_stages=[2, 3],\n"
            "        threads=[128, 256],\n"
            "    )\n"
            "    return [\n"
            "        dict(zip(iter_params, values))\n"
            "        for values in itertools.product(*iter_params.values())\n"
            "    ]\n"
            "\n"
            "\n"
            "@tilelang.autotune(configs=get_configs(), cache_input_tensors=True)\n"
            "@tilelang.jit(out_idx=[-1])\n"
            "def matmul(\n"
            "    M, N, K,\n"
            "    block_M: int,\n"
            "    block_N: int,\n"
            "    block_K: int,\n"
            "    num_stages: int,\n"
            "    threads: int,\n"
            "):\n"
            "    dtype = T.float16\n"
            "    accum_dtype = T.float32\n"
            "\n"
            "    @T.prim_func\n"
            "    def gemm(\n"
            "        A: T.Tensor((M, K), dtype),\n"
            "        B: T.Tensor((K, N), dtype),\n"
            "        C: T.Tensor((M, N), dtype),\n"
            "    ):\n"
            "        with T.Kernel(\n"
            "            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads\n"
            "        ) as (bx, by):\n"
            "            A_shared = T.alloc_shared((block_M, block_K), dtype)\n"
            "            B_shared = T.alloc_shared((block_K, block_N), dtype)\n"
            "            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)\n"
            "\n"
            "            T.use_swizzle(panel_size=10, enable=True)\n"
            "            T.clear(C_local)\n"
            "\n"
            "            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):\n"
            "                T.copy(A[by * block_M, k * block_K], A_shared)\n"
            "                T.copy(B[k * block_K, bx * block_N], B_shared)\n"
            "                T.gemm(A_shared, B_shared, C_local)\n"
            "\n"
            "            T.copy(C_local, C[by * block_M, bx * block_N])\n"
            "\n"
            "    return gemm\n"
        ),
    },
}


# ---------------------------------------------------------------------------
# Optimization Tips (used by MCP resources)
# ---------------------------------------------------------------------------
OPTIMIZATION_TIPS: dict[str, dict] = {
    "pipeline": {
        "topic": "Software Pipelining",
        "tips": [
            "Use `T.Pipelined(iters, num_stages=N)` to overlap memory loads with compute.",
            "Typical `num_stages` values: 2 (minimal overlap), 3 (good balance), 4 (aggressive, more shared memory).",
            "More stages = more shared memory usage. If shared memory is tight, reduce `num_stages`.",
            "Pipeline is most effective for memory-bound kernels (e.g., GEMM inner loop over K).",
            "Use `num_stages=0` to disable pipelining (useful for debugging or when compute-bound).",
            "On Hopper (SM90+), pipelining can auto-lower to TMA + warp-specialization for best overlap.",
        ],
        "example": (
            "# Good: pipelined K-loop overlaps copy and compute\n"
            "for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):\n"
            "    T.copy(A[by * BM, ko * BK], A_shared)\n"
            "    T.copy(B[ko * BK, bx * BN], B_shared)\n"
            "    T.gemm(A_shared, B_shared, C_local)\n"
            "\n"
            "# Bad: serial loop, no overlap\n"
            "for ko in T.serial(T.ceildiv(K, block_K)):\n"
            "    T.copy(A[by * BM, ko * BK], A_shared)\n"
            "    T.copy(B[ko * BK, bx * BN], B_shared)\n"
            "    T.gemm(A_shared, B_shared, C_local)\n"
        ),
    },
    "swizzle": {
        "topic": "Threadblock Swizzle (L2 Cache Locality)",
        "tips": [
            "Use `T.use_swizzle(panel_size=10)` to remap threadblock scheduling order.",
            "Improves L2 cache hit rate by making adjacent threadblocks access nearby data.",
            "Essential for large GEMM and attention kernels where data reuse across blocks matters.",
            "`panel_size` controls the width of the swizzled group — typical values: 7-10.",
            "Use `order='row'` (default) for row-major data, `order='col'` for column-major.",
            "Almost always improves performance; virtually no downside to enabling it.",
        ],
        "example": (
            "with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):\n"
            "    # Enable threadblock swizzle — one line, big impact on L2 locality\n"
            "    T.use_swizzle(panel_size=10, enable=True)\n"
            "    # ... rest of kernel\n"
        ),
    },
    "shared_memory_layout": {
        "topic": "Shared Memory Layout (Bank Conflict Reduction)",
        "tips": [
            "Use `T.annotate_layout` with swizzled layouts to avoid shared memory bank conflicts.",
            "Bank conflicts serialize parallel memory accesses within a warp — a major perf killer.",
            "TileLang provides `make_swizzled_layout()` to auto-generate conflict-free layouts.",
            "For MMA/WMMA operations, use `make_mma_swizzle_layout()` from `tilelang.intrinsics`.",
            "Apply to both A_shared and B_shared in GEMM kernels for best effect.",
        ],
        "example": (
            "from tilelang.intrinsics import make_mma_swizzle_layout\n"
            "\n"
            "A_shared = T.alloc_shared((block_M, block_K), dtype)\n"
            "B_shared = T.alloc_shared((block_K, block_N), dtype)\n"
            "\n"
            "# Annotate layouts to avoid bank conflicts\n"
            "T.annotate_layout({\n"
            "    A_shared: make_mma_swizzle_layout(A_shared),\n"
            "    B_shared: make_mma_swizzle_layout(B_shared),\n"
            "})\n"
        ),
    },
    "warp_policy": {
        "topic": "GEMM Warp Policy",
        "tips": [
            "`T.GemmWarpPolicy.Square` (default): balanced warp partition across M and N.",
            "`T.GemmWarpPolicy.FullRow`: all warps span full rows — best for attention (S@V pattern).",
            "`T.GemmWarpPolicy.FullCol`: all warps span full columns — good for tall-skinny matrices.",
            "Choose based on which dimension is larger or which result needs to stay in registers.",
            "For attention: use `FullRow` so each warp holds a full row of attention scores.",
            "For MLA / multi-head attention with split-K: `FullCol` can be more efficient.",
            "Custom partition: `T.GemmWarpPolicy(m_warp=2, n_warp=4)` for explicit control.",
        ],
        "example": (
            "# Attention pattern: Q @ K^T with FullRow so each warp owns a row of scores\n"
            "T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,\n"
            "       policy=T.GemmWarpPolicy.FullRow)\n"
            "\n"
            "# ... softmax on acc_s ...\n"
            "\n"
            "# Then acc_s @ V also with FullRow to keep accumulator layout consistent\n"
            "T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)\n"
        ),
    },
    "memory_coalescing": {
        "topic": "Memory Coalescing",
        "tips": [
            "Use `T.copy()` for data movement — it auto-coalesces and vectorizes loads.",
            "For manual loops, use `T.Parallel` to distribute work evenly across threads.",
            "Set `coalesced_width=N` on `T.copy()` to explicitly control vectorization width.",
            "Typical `coalesced_width` values: 4-8 for fp16, 2-4 for fp32.",
            "Ensure the innermost dimension is contiguous in memory for best coalescing.",
            "Prefer `T.copy(A[offset, :], A_shared)` over manual element-by-element loops.",
        ],
        "example": (
            "# Good: T.copy auto-coalesces\n"
            "T.copy(A[by * BM, ko * BK], A_shared)\n"
            "\n"
            "# Good: explicit coalesced_width for fine control\n"
            "T.copy(K[bz, k * BN, by, :], K_shared, coalesced_width=8)\n"
            "\n"
            "# OK but less optimal: manual Parallel loop\n"
            "for i, j in T.Parallel(block_M, block_K):\n"
            "    A_shared[i, j] = A[by * block_M + i, ko * block_K + j]\n"
        ),
    },
    "autotune": {
        "topic": "Autotuning",
        "tips": [
            "Use `@tilelang.autotune(configs=get_configs())` to search over block sizes, stages, threads.",
            "Stack with `@tilelang.jit(out_idx=[...])` — autotune goes on top.",
            "Define configs with `itertools.product` over key parameters.",
            "Key tuning parameters: block_M, block_N, block_K, num_stages, threads.",
            "Use `cache_input_tensors=True` for faster tuning (avoids re-allocating inputs).",
            "Access best config via `kernel.config` after tuning.",
            "For attention: also tune `coalesced_width` and `panel_size`.",
            "Start with a coarse grid, then refine around the best config.",
        ],
        "example": (
            "import itertools\n"
            "\n"
            "def get_configs():\n"
            "    params = dict(\n"
            "        block_M=[64, 128, 256],\n"
            "        block_N=[64, 128, 256],\n"
            "        block_K=[32, 64],\n"
            "        num_stages=[2, 3],\n"
            "        threads=[128, 256],\n"
            "    )\n"
            "    return [dict(zip(params, v)) for v in itertools.product(*params.values())]\n"
            "\n"
            "@tilelang.autotune(configs=get_configs(), cache_input_tensors=True)\n"
            "@tilelang.jit(out_idx=[-1])\n"
            "def my_kernel(M, N, K, block_M: int, block_N: int, block_K: int,\n"
            "              num_stages: int, threads: int):\n"
            "    # ... kernel definition ...\n"
            "    pass\n"
        ),
    },
    "tiling": {
        "topic": "Tile Size Selection",
        "tips": [
            "block_M, block_N: larger tiles = more data reuse, but more shared memory and registers.",
            "block_K: affects inner loop trip count — larger K means fewer pipeline iterations.",
            "Common GEMM tile sizes: 128x128x32, 128x256x64, 256x128x32.",
            "Tile sizes must be multiples of the tensor core shape (16x16 for fp16 on NVIDIA).",
            "Too-large tiles can spill registers or exceed shared memory — causes massive slowdown.",
            "Use autotune to find optimal tile sizes for your specific problem shape and GPU.",
            "For small problems: use smaller tiles to fill more SMs.",
            "For large problems: use larger tiles for better data reuse.",
        ],
    },
    "occupancy": {
        "topic": "GPU Occupancy",
        "tips": [
            "Occupancy = active warps / max warps per SM. Higher is generally better but not always.",
            "`threads=128` = 4 warps, `threads=256` = 8 warps per thread block.",
            "More threads per block = fewer blocks can run simultaneously (shared mem and register limits).",
            "Balance: enough threads to hide latency, not so many that you run out of resources.",
            "Pipeline stages increase shared memory usage, which can limit occupancy.",
            "Use the profiler to measure: `kernel.get_profiler().do_bench()` returns latency in ms.",
            "If your kernel is compute-bound, lower occupancy with more registers can be faster.",
        ],
    },
    "attention": {
        "topic": "Flash Attention Optimization",
        "tips": [
            "Use online softmax (running max + log-sum-exp) to avoid materializing the full S matrix.",
            "Key pattern: Q_shared stays in shared, stream K and V blocks through the pipeline.",
            "Use `T.GemmWarpPolicy.FullRow` for both Q@K^T and S@V GEMM operations.",
            "Use `T.exp2()` instead of `T.exp()` — it maps to a single PTX instruction.",
            "Pre-scale Q by `1/sqrt(d) * log2(e)` to fuse the scale into exp2.",
            "For causal attention: compute `loop_range` to skip unnecessary K blocks.",
            "Allocate `scores_max`, `scores_sum`, `logsum` as 1D fragments of shape [block_M].",
            "Cast attention scores to fp16 before the S@V GEMM to use tensor cores.",
        ],
        "example": (
            "# Flash Attention core pattern:\n"
            "scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) scaling\n"
            "\n"
            "for k in T.Pipelined(loop_range, num_stages=num_stages):\n"
            "    T.copy(K[...], K_shared)\n"
            "\n"
            "    # Q @ K^T -> acc_s (attention scores)\n"
            "    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,\n"
            "           policy=T.GemmWarpPolicy.FullRow)\n"
            "\n"
            "    # Online softmax update\n"
            "    T.copy(scores_max, scores_max_prev)\n"
            "    T.reduce_max(acc_s, scores_max, dim=1, clear=False)\n"
            "    for i in T.Parallel(block_M):\n"
            "        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)\n"
            "    for i, j in T.Parallel(block_M, block_N):\n"
            "        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)\n"
            "\n"
            "    # Rescale previous partial output\n"
            "    for i, j in T.Parallel(block_M, dim):\n"
            "        acc_o[i, j] *= scores_scale[i]\n"
            "\n"
            "    # S @ V -> accumulate into acc_o\n"
            "    T.copy(acc_s, acc_s_cast)  # fp32 -> fp16\n"
            "    T.copy(V[...], V_shared)\n"
            "    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)\n"
        ),
    },
    "data_types": {
        "topic": "Data Type Selection",
        "tips": [
            "Use fp16 (`T.float16`) for inputs/outputs — half the memory bandwidth of fp32.",
            "Accumulate in fp32 (`T.float32`) for numerical stability in GEMM and reductions.",
            "Use `T.bfloat16` for training workloads — wider dynamic range than fp16.",
            "FP8 (`T.float8_e4m3fn`, `T.float8_e5m2`) for inference on Hopper+ — 2x throughput.",
            "Cast only at boundaries: keep intermediate results in higher precision.",
            "For attention scores: compute in fp32, cast to fp16 only before the S@V GEMM.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Best Practices (used by MCP resources)
# ---------------------------------------------------------------------------
BEST_PRACTICES: list[dict] = [
    {
        "category": "Memory Hierarchy",
        "practices": [
            "Follow Global -> Shared -> Fragment/Register -> Shared -> Global data flow.",
            "Use `T.alloc_shared` for inter-thread data, `T.alloc_fragment` for per-thread accumulators.",
            "Minimize global memory accesses — maximize data reuse in shared memory.",
            "Use `T.copy()` instead of manual loops for data movement (auto-optimized).",
        ],
    },
    {
        "category": "Compute",
        "practices": [
            "Use `T.gemm()` for matrix multiply — it maps to tensor cores automatically.",
            "Always `T.clear(accumulator)` before the compute loop.",
            "Prefer `T.exp2` over `T.exp` (single PTX instruction).",
            "Use `T.reduce_max/sum` for reductions instead of manual loops.",
        ],
    },
    {
        "category": "Loop Structure",
        "practices": [
            "Use `T.Pipelined` for the main compute loop (overlap memory and compute).",
            "Use `T.Parallel` for elementwise operations (auto-parallelized across threads).",
            "Use `T.serial` only for small sequential loops or loop-carried dependencies.",
            "Use `T.unroll` for small constant-trip-count inner loops.",
        ],
    },
    {
        "category": "Performance Annotations",
        "practices": [
            "Always enable `T.use_swizzle(panel_size=10)` for GEMM and attention kernels.",
            "Use `T.annotate_layout` with swizzled layouts to avoid bank conflicts.",
            "Set `coalesced_width` on `T.copy` when default vectorization isn't optimal.",
            "Use `eviction_policy='evict_first'` for streaming data not reused across iterations.",
        ],
    },
    {
        "category": "Autotuning",
        "practices": [
            "Always autotune tile sizes (block_M, block_N, block_K) — optimal values vary by GPU and shape.",
            "Include `num_stages` and `threads` in the autotune search space.",
            "Use `cache_input_tensors=True` for faster tuning iterations.",
            "Start with a coarse grid, refine around the best configuration.",
        ],
    },
    {
        "category": "Common Pitfalls",
        "practices": [
            "Don't forget `T.clear()` before accumulation loops — uninitialized fragments cause wrong results.",
            "Tile sizes must be multiples of tensor core shapes (16 for fp16 on NVIDIA).",
            "Too many pipeline stages can exhaust shared memory — reduce `num_stages` if compilation fails.",
            "Avoid reading from a `T.alloc_fragment` buffer after writing it via `T.gemm` without `T.copy` to shared first, if cross-thread data sharing is needed.",
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════
#  Public API — LSP functions
# ═══════════════════════════════════════════════════════════════════


def get_t_completions() -> list[TileLangSymbol]:
    """Return all symbols available under the T. namespace."""
    return list(_SYMBOLS)


def get_tilelang_completions() -> list[TileLangSymbol]:
    """Return all symbols available under the tilelang. namespace."""
    return list(_TILELANG_SYMBOLS)


def get_snippets() -> dict:
    """Return snippet definitions."""
    return dict(_SNIPPETS)


def lookup_symbol(name: str) -> TileLangSymbol | None:
    """Look up a symbol by name across all namespaces."""
    return _BY_NAME.get(name)


# ═══════════════════════════════════════════════════════════════════
#  Public API — MCP functions
# ═══════════════════════════════════════════════════════════════════


def list_symbols_by_category(category: str) -> list[dict]:
    """List all symbols in a category (returns dicts for MCP compatibility)."""
    syms = _BY_CATEGORY.get(category, [])
    return [
        {"name": s.name, "kind": s.kind, "detail": s.detail,
         "documentation": s.documentation, "category": s.category}
        for s in syms
    ]


def list_categories() -> dict[str, str]:
    """Return all categories with descriptions."""
    return dict(CATEGORIES)


def get_template(name: str) -> Optional[dict]:
    """Get a kernel template by name."""
    return TEMPLATES.get(name)


def list_templates() -> list[str]:
    """List all template names."""
    return list(TEMPLATES.keys())


def get_optimization_tip(topic: str) -> Optional[dict]:
    """Get optimization tips for a topic."""
    return OPTIMIZATION_TIPS.get(topic)


def list_optimization_topics() -> list[str]:
    """List all optimization topics."""
    return list(OPTIMIZATION_TIPS.keys())


def get_best_practices() -> list[dict]:
    """Return all best practices."""
    return BEST_PRACTICES
