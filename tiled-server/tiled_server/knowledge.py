"""TileLang DSL knowledge base — completions, docs, and signatures."""

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
    deprecated: bool = False


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
    ),

    # -- Buffer types --
    TileLangSymbol(
        "Buffer", "type", "T.Buffer",
        "General buffer type for kernel parameters.",
    ),
    TileLangSymbol(
        "SharedBuffer", "type", "T.SharedBuffer",
        "Buffer in shared memory scope.",
    ),
    TileLangSymbol(
        "LocalBuffer", "type", "T.LocalBuffer",
        "Buffer in local (thread-private) memory scope.",
    ),
    TileLangSymbol(
        "FragmentBuffer", "type", "T.FragmentBuffer",
        "Buffer for tensor core fragment storage.",
    ),

    # -- Memory allocation --
    TileLangSymbol(
        "alloc_shared", "function",
        "T.alloc_shared(shape, dtype) -> Buffer",
        "Allocate a shared memory buffer for inter-thread communication.\n\n"
        "```python\n"
        "A_shared = T.alloc_shared((block_M, block_K), T.float16)\n"
        "```",
        snippet="alloc_shared((${1:block_M}, ${2:block_K}), ${3:dtype})",
    ),
    TileLangSymbol(
        "alloc_fragment", "function",
        "T.alloc_fragment(shape, dtype) -> Buffer",
        "Allocate a fragment buffer for register-level / tensor core storage.\n\n"
        "```python\n"
        "C_local = T.alloc_fragment((block_M, block_N), T.float32)\n"
        "```",
        snippet="alloc_fragment((${1:block_M}, ${2:block_N}), ${3:dtype})",
    ),
    TileLangSymbol(
        "alloc_local", "function",
        "T.alloc_local(shape, dtype) -> Buffer",
        "Allocate thread-private local memory.\n\n"
        "```python\n"
        "tmp = T.alloc_local((4,), T.float32)\n"
        "```",
        snippet="alloc_local((${1:shape},), ${2:dtype})",
    ),
    TileLangSymbol(
        "alloc_var", "function",
        "T.alloc_var(dtype) -> Buffer",
        "Allocate a single-element variable buffer.",
        snippet="alloc_var(${1:dtype})",
    ),
    TileLangSymbol(
        "alloc_barrier", "function",
        "T.alloc_barrier() -> Buffer",
        "Allocate a barrier for synchronization.",
    ),
    TileLangSymbol(
        "alloc_reducer", "function",
        "T.alloc_reducer(dtype) -> Buffer",
        "Allocate a reducer for reduction operations.",
    ),
    TileLangSymbol(
        "alloc_tmem", "function",
        "T.alloc_tmem(shape, dtype) -> Buffer",
        "Allocate tensor memory (SM100+).",
        snippet="alloc_tmem((${1:shape},), ${2:dtype})",
    ),
    TileLangSymbol(
        "empty", "function",
        "T.empty(shape, dtype) -> Buffer",
        "Allocate an uninitialized buffer.",
        snippet="empty((${1:shape},), ${2:dtype})",
    ),

    # -- Data movement --
    TileLangSymbol(
        "copy", "function",
        "T.copy(src, dst)",
        "Copy data between buffers or between global/shared memory.\n\n"
        "```python\n"
        "T.copy(A[by * block_M, k * block_K], A_shared)\n"
        "T.copy(C_local, C[by * block_M, bx * block_N])\n"
        "```",
        snippet="copy(${1:src}, ${2:dst})",
    ),
    TileLangSymbol(
        "async_copy", "function",
        "T.async_copy(src, dst)",
        "Asynchronous copy (cp.async) between global and shared memory.",
        snippet="async_copy(${1:src}, ${2:dst})",
    ),

    # -- Compute operations --
    TileLangSymbol(
        "gemm", "function",
        "T.gemm(A, B, C, policy=None)",
        "Perform matrix multiplication C += A @ B using tensor cores.\n\n"
        "```python\n"
        "T.gemm(A_shared, B_shared, C_local)\n"
        "```",
        snippet="gemm(${1:A_shared}, ${2:B_shared}, ${3:C_local})",
    ),
    TileLangSymbol(
        "fill", "function",
        "T.fill(buffer, value)",
        "Fill a buffer with a constant value.",
        snippet="fill(${1:buffer}, ${2:0})",
    ),
    TileLangSymbol(
        "clear", "function",
        "T.clear(buffer)",
        "Clear (zero-initialize) a buffer.\n\n"
        "```python\n"
        "T.clear(C_local)\n"
        "```",
        snippet="clear(${1:buffer})",
    ),

    # -- Loop constructs --
    TileLangSymbol(
        "Pipelined", "function",
        "T.Pipelined(extent, num_stages=N)",
        "Create a software-pipelined loop.\n\n"
        "```python\n"
        "for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):\n"
        "    ...\n"
        "```",
        snippet="Pipelined(${1:extent}, num_stages=${2:3})",
    ),
    TileLangSymbol(
        "Parallel", "function",
        "T.Parallel(*extents)",
        "Create nested parallel loops for element-wise operations.\n\n"
        "```python\n"
        "for i, j in T.Parallel(block_M, block_N):\n"
        "    C[i, j] = A[i, j] + B[i, j]\n"
        "```",
        snippet="Parallel(${1:block_M}, ${2:block_N})",
    ),
    TileLangSymbol(
        "Persistent", "function",
        "T.Persistent(*extents)",
        "Create a persistent loop that keeps a thread block alive across iterations.",
        snippet="Persistent(${1:extent})",
    ),
    TileLangSymbol(
        "serial", "function",
        "T.serial(start, stop)",
        "Create a serial (sequential) for loop.",
        snippet="serial(${1:0}, ${2:N})",
    ),
    TileLangSymbol(
        "unroll", "function",
        "T.unroll(start, stop)",
        "Create a fully unrolled for loop.",
        snippet="unroll(${1:0}, ${2:N})",
    ),
    TileLangSymbol(
        "vectorized", "function",
        "T.vectorized(start, stop)",
        "Create a vectorized for loop.",
        snippet="vectorized(${1:0}, ${2:N})",
    ),

    # -- Reduction --
    TileLangSymbol(
        "reduce_sum", "function",
        "T.reduce_sum(src, dst, dim=None)",
        "Perform a sum reduction.",
        snippet="reduce_sum(${1:src}, ${2:dst}, dim=${3:0})",
    ),
    TileLangSymbol(
        "reduce_max", "function",
        "T.reduce_max(src, dst, dim=None)",
        "Perform a max reduction.",
        snippet="reduce_max(${1:src}, ${2:dst}, dim=${3:0})",
    ),
    TileLangSymbol(
        "reduce_min", "function",
        "T.reduce_min(src, dst, dim=None)",
        "Perform a min reduction.",
        snippet="reduce_min(${1:src}, ${2:dst}, dim=${3:0})",
    ),
    TileLangSymbol(
        "reduce_abssum", "function",
        "T.reduce_abssum(src, dst, dim=None)",
        "Perform an absolute-value sum reduction.",
    ),
    TileLangSymbol(
        "reduce_absmax", "function",
        "T.reduce_absmax(src, dst, dim=None)",
        "Perform an absolute-value max reduction.",
    ),
    TileLangSymbol(
        "reduce", "function",
        "T.reduce(op, src, dst, dim=None)",
        "Generic reduction with a given binary operator.",
    ),
    TileLangSymbol(
        "cumsum", "function",
        "T.cumsum(src, dst, dim=None)",
        "Compute cumulative sum along a dimension.",
    ),
    TileLangSymbol(
        "finalize_reducer", "function",
        "T.finalize_reducer(reducer)",
        "Finalize a reducer after reduction iterations.",
    ),
    TileLangSymbol(
        "warp_reduce_sum", "function",
        "T.warp_reduce_sum(value)",
        "Warp-level sum reduction.",
    ),
    TileLangSymbol(
        "warp_reduce_max", "function",
        "T.warp_reduce_max(value)",
        "Warp-level max reduction.",
    ),
    TileLangSymbol(
        "warp_reduce_min", "function",
        "T.warp_reduce_min(value)",
        "Warp-level min reduction.",
    ),

    # -- Atomic operations --
    TileLangSymbol(
        "atomic_add", "function",
        "T.atomic_add(dst, value)",
        "Atomically add a value to a memory location.",
        snippet="atomic_add(${1:dst}, ${2:value})",
    ),
    TileLangSymbol(
        "atomic_max", "function",
        "T.atomic_max(dst, value)",
        "Atomically compute max.",
    ),
    TileLangSymbol(
        "atomic_min", "function",
        "T.atomic_min(dst, value)",
        "Atomically compute min.",
    ),

    # -- Utility --
    TileLangSymbol(
        "ceildiv", "function",
        "T.ceildiv(a, b) -> int",
        "Compute ceiling division: `(a + b - 1) // b`.",
        snippet="ceildiv(${1:a}, ${2:b})",
    ),
    TileLangSymbol(
        "print", "function",
        "T.print(fmt, *args)",
        "Device-side printf for debugging.\n\n"
        "```python\n"
        'T.print(\"val = {}\", val)\n'
        "```",
    ),
    TileLangSymbol(
        "device_assert", "function",
        "T.device_assert(cond, msg)",
        "Assert a condition on the device side.",
    ),

    # -- Memory annotations --
    TileLangSymbol(
        "use_swizzle", "function",
        "T.use_swizzle(nbit, enable=True)",
        "Annotate a shared buffer to use a swizzle pattern for bank conflict avoidance.",
        snippet="use_swizzle(${1:nbit})",
    ),
    TileLangSymbol(
        "annotate_layout", "function",
        "T.annotate_layout(buffer, layout)",
        "Annotate a buffer with a custom layout.",
    ),

    # -- Thread/block queries --
    TileLangSymbol(
        "get_thread_binding", "function",
        "T.get_thread_binding() -> int",
        "Get the current thread index within the block (threadIdx.x).",
    ),
    TileLangSymbol(
        "get_thread_bindings", "function",
        "T.get_thread_bindings() -> tuple",
        "Get all thread binding indices.",
    ),
    TileLangSymbol(
        "get_block_binding", "function",
        "T.get_block_binding() -> int",
        "Get the current block index.",
    ),
    TileLangSymbol(
        "get_block_bindings", "function",
        "T.get_block_bindings() -> tuple",
        "Get all block binding indices.",
    ),

    # -- Misc --
    TileLangSymbol(
        "reshape", "function",
        "T.reshape(buffer, new_shape)",
        "Reshape a buffer to a new shape.",
        snippet="reshape(${1:buffer}, (${2:new_shape},))",
    ),
    TileLangSymbol(
        "view", "function",
        "T.view(buffer, new_shape)",
        "View a buffer with a new shape (no copy).",
    ),
    TileLangSymbol(
        "clamp", "function",
        "T.clamp(value, min_val, max_val)",
        "Clamp a value between min and max.",
    ),
    TileLangSymbol(
        "dp4a", "function",
        "T.dp4a(a, b, c)",
        "4-element dot product with accumulate (int8).",
    ),
    TileLangSymbol(
        "loop_break", "function",
        "T.loop_break()",
        "Break out of the current loop.",
    ),
    TileLangSymbol(
        "any_of", "function",
        "T.any_of(*args)",
        "Logical OR across arguments.",
    ),
    TileLangSymbol(
        "all_of", "function",
        "T.all_of(*args)",
        "Logical AND across arguments.",
    ),
    TileLangSymbol(
        "dynamic", "function",
        "T.dynamic(name)",
        "Declare a dynamic (runtime) symbolic variable.",
    ),
    TileLangSymbol(
        "symbolic", "function",
        "T.symbolic(name)",
        "Declare a compile-time symbolic variable.",
    ),
    TileLangSymbol(
        "import_source", "function",
        "T.import_source(source)",
        "Import raw C source code into the kernel.",
    ),
    TileLangSymbol(
        "make_tensor", "function",
        "T.make_tensor(ptr, shape, dtype)",
        "Create a tensor from a raw pointer.",
    ),
    TileLangSymbol(
        "ptr", "function",
        "T.ptr(dtype)",
        "Create a pointer type annotation.",
    ),
    TileLangSymbol(
        "index_to_coordinates", "function",
        "T.index_to_coordinates(index, shape)",
        "Convert a flat index to multi-dimensional coordinates.",
    ),

    # -- Random --
    TileLangSymbol(
        "rng_init", "function",
        "T.rng_init(seed)",
        "Initialize the random number generator state.",
    ),
    TileLangSymbol(
        "rng_rand", "function",
        "T.rng_rand(state)",
        "Generate a random integer.",
    ),
    TileLangSymbol(
        "rng_rand_float", "function",
        "T.rng_rand_float(state)",
        "Generate a random float in [0, 1).",
    ),

    # -- Cluster --
    TileLangSymbol(
        "cluster_sync", "function",
        "T.cluster_sync()",
        "Synchronize all blocks in a cluster.",
    ),
    TileLangSymbol(
        "cluster_arrive", "function",
        "T.cluster_arrive()",
        "Signal cluster arrival.",
    ),
    TileLangSymbol(
        "cluster_wait", "function",
        "T.cluster_wait()",
        "Wait for cluster arrival.",
    ),
    TileLangSymbol(
        "block_rank_in_cluster", "function",
        "T.block_rank_in_cluster()",
        "Get the rank of the current block within its cluster.",
    ),

    # -- PDL --
    TileLangSymbol(
        "pdl_trigger", "function",
        "T.pdl_trigger()",
        "Trigger a PDL (Programmatic Dependent Launch) event.",
    ),
    TileLangSymbol(
        "pdl_sync", "function",
        "T.pdl_sync()",
        "Synchronize with PDL events.",
    ),

    # -- Warp specialization --
    TileLangSymbol(
        "ws", "class",
        "T.ws",
        "Warp specialization utilities. Access via `T.ws.*`.",
    ),

    # -- Layout --
    TileLangSymbol(
        "Layout", "class",
        "T.Layout(mapping, shape)",
        "Define a data layout for buffer or loop mapping.",
    ),
    TileLangSymbol(
        "Fragment", "class",
        "T.Fragment(mapping, shape)",
        "Define a fragment layout for tensor core operations.",
    ),

    # -- Load/store intrinsics --
    TileLangSymbol("__ldg", "function", "T.__ldg(ptr)", "Load through texture cache (read-only)."),
    TileLangSymbol("ldg32", "function", "T.ldg32(ptr)", "32-bit load through texture cache."),
    TileLangSymbol("ldg64", "function", "T.ldg64(ptr)", "64-bit load through texture cache."),
    TileLangSymbol("ldg128", "function", "T.ldg128(ptr)", "128-bit load through texture cache."),
    TileLangSymbol("ldg256", "function", "T.ldg256(ptr)", "256-bit load through texture cache."),
    TileLangSymbol("stg32", "function", "T.stg32(ptr, val)", "32-bit global store."),
    TileLangSymbol("stg64", "function", "T.stg64(ptr, val)", "64-bit global store."),
    TileLangSymbol("stg128", "function", "T.stg128(ptr, val)", "128-bit global store."),
    TileLangSymbol("stg256", "function", "T.stg256(ptr, val)", "256-bit global store."),

    # -- GemmWarpPolicy --
    TileLangSymbol(
        "GemmWarpPolicy", "class",
        "T.GemmWarpPolicy",
        "Warp policy configuration for gemm operations.",
    ),

    # -- Kernel launch frame --
    TileLangSymbol(
        "KernelLaunchFrame", "class",
        "T.KernelLaunchFrame",
        "Low-level kernel launch frame (advanced usage).",
    ),
]

# Build dtype symbols
for _dt in _SCALAR_DTYPES:
    _SYMBOLS.append(TileLangSymbol(
        _dt, "constant", f"T.{_dt}",
        f"Data type: `{_dt}`",
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
        "- `out_idx`: Indices of output tensors (negative = from end).",
        snippet="jit(out_idx=[${1:-1}])",
    ),
    TileLangSymbol(
        "compile", "function",
        "tilelang.compile(func, ...)",
        "Compile a TileLang function without JIT.",
    ),
    TileLangSymbol(
        "autotune", "function",
        "tilelang.autotune(func, configs, ...)",
        "Auto-tune a kernel over a set of configurations.",
    ),
    TileLangSymbol(
        "Profiler", "class",
        "tilelang.Profiler",
        "Profiler for benchmarking compiled kernels.",
    ),
]

# ---------------------------------------------------------------------------
# Snippet templates for common patterns
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
    for sym in _SYMBOLS:
        if sym.name == name:
            return sym
    for sym in _TILELANG_SYMBOLS:
        if sym.name == name:
            return sym
    return None
