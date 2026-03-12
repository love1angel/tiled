"""TileLang MCP server — real GPU tools + static knowledge resources.

Tools (5): compile_and_benchmark, analyze_kernel, autotune, search_examples,
read_example — wrap tilelang's Analyzer, Profiler, and AutoTuner.

Resources: API docs, templates, optimization tips, best practices — served as
static context (no round-trip needed for the LLM to read them).
"""

from __future__ import annotations

import importlib.util
import os
import re
from typing import Any

from mcp.server.fastmcp import FastMCP

from .knowledge import (
    OPTIMIZATION_TIPS,
    lookup_symbol,
    list_symbols_by_category,
    list_categories,
    get_template,
    list_templates,
    get_optimization_tip,
    list_optimization_topics,
    get_best_practices,
)

mcp = FastMCP(
    "tilelang-mcp",
    instructions=(
        "TileLang GPU kernel assistant.  Use the 5 tools for real GPU work "
        "(compile_and_benchmark, analyze_kernel, autotune, search_examples, "
        "read_example).  Read the resources for API docs, templates, "
        "optimization tips, and best practices — they are static context."
    ),
)


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def _check_tilelang():
    """Return (tilelang_module, None) or (None, error_string)."""
    try:
        import tilelang
        return tilelang, None
    except ImportError:
        return None, "TileLang is not installed. Install with: pip install tilelang"


def _check_gpu():
    """Return (is_available: bool, description: str)."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"cuda ({torch.cuda.get_device_name(0)})"
    except ImportError:
        pass
    return False, "none"


_DANGEROUS_PATTERNS = [
    r'\bos\.system\b', r'\bsubprocess\b', r'\b__import__\b',
    r'\beval\b', r'\bexec\b', r'\bopen\b.*["\']w',
    r'\bshutil\.(rmtree|move)\b', r'\bos\.remove\b',
]


def _check_code_safety(code: str) -> str | None:
    """Return rejection message or None if safe."""
    for pat in _DANGEROUS_PATTERNS:
        if re.search(pat, code):
            return f"Rejected: code contains potentially unsafe pattern matching `{pat}`."
    return None


def _find_tilelang_root() -> str | None:
    """Find tilelang package/source root."""
    try:
        spec = importlib.util.find_spec("tilelang")
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            if os.path.isdir(os.path.join(pkg_dir, "language")):
                return os.path.dirname(pkg_dir)
    except (ModuleNotFoundError, ValueError):
        pass
    src = os.path.join(os.path.expanduser("~"), "tilelang")
    if os.path.isdir(os.path.join(src, "tilelang", "language")):
        return src
    return None


# ═══════════════════════════════════════════════════════════════════
#  Resources  (static knowledge — served as context, zero tool calls)
# ═══════════════════════════════════════════════════════════════════


@mcp.resource("tilelang://api/index")
def api_index() -> str:
    """List all TileLang API categories and their symbols."""
    lines = ["# TileLang API Reference\n"]
    for cat, desc in list_categories().items():
        lines.append(f"## {cat} — {desc}\n")
        for sym in list_symbols_by_category(cat):
            lines.append(f"- `{sym['detail']}` — {sym['documentation'].split(chr(10))[0]}")
        lines.append("")
    return "\n".join(lines)


@mcp.resource("tilelang://api/{name}")
def api_lookup(name: str) -> str:
    """Look up documentation for a specific TileLang API symbol."""
    clean = re.sub(r"^(T|tilelang)\.", "", name.strip())
    sym = lookup_symbol(clean)
    if not sym:
        return f"Symbol '{name}' not found."
    return f"## {sym.detail}\n\n**Kind:** {sym.kind}\n\n{sym.documentation}"


@mcp.resource("tilelang://templates")
def templates_index() -> str:
    """All available TileLang kernel code templates."""
    lines = ["# TileLang Kernel Templates\n"]
    for tname in list_templates():
        tmpl = get_template(tname)
        lines.append(f"## {tmpl['name']}\n")
        lines.append(f"{tmpl['description']}\n")
        lines.append(f"```python\n{tmpl['code']}\n```\n")
    return "\n".join(lines)


@mcp.resource("tilelang://optimization-tips")
def optimization_tips_resource() -> str:
    """Performance optimization tips for TileLang kernels."""
    lines = ["# TileLang Optimization Tips\n"]
    for topic in list_optimization_topics():
        tip = get_optimization_tip(topic)
        if not tip:
            continue
        lines.append(f"## {tip['topic']}\n")
        for t in tip.get("tips", []):
            lines.append(f"- {t}")
        if tip.get("example"):
            lines.append(f"\n```python\n{tip['example']}\n```")
        lines.append("")
    return "\n".join(lines)


@mcp.resource("tilelang://best-practices")
def best_practices_resource() -> str:
    """TileLang best practices and common patterns."""
    lines = ["# TileLang Best Practices\n"]
    for bp in get_best_practices():
        lines.append(f"## {bp.get('category', 'General')}\n")
        for p in bp.get("practices", []):
            lines.append(f"- {p}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Tool 1: compile_and_benchmark
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def compile_and_benchmark(code: str) -> str:
    """Compile a TileLang kernel and benchmark it on GPU.

    Takes complete Python source code that defines a TileLang kernel,
    compiles it using ``tilelang.compile()``, and benchmarks real GPU
    performance using tilelang's Profiler with L2 cache flushing.

    Also runs static analysis via tilelang's Analyzer to report FLOPs,
    memory bandwidth, and roofline efficiency.

    The code must assign a compiled kernel to ``kernel``.  Optionally
    assign the ``@T.prim_func`` to ``func`` for static analysis.

    Example::

        import tilelang
        import tilelang.language as T

        M, N, K = 1024, 1024, 1024
        block_M, block_N, block_K = 128, 128, 32

        @T.prim_func
        def func(A: T.Tensor((M, K), T.float16),
                 B: T.Tensor((K, N), T.float16),
                 C: T.Tensor((M, N), T.float16)):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), T.float16)
                B_s = T.alloc_shared((block_K, block_N), T.float16)
                C_f = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_f)
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(B[ko * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_f)
                T.copy(C_f, C[by * block_M, bx * block_N])

        kernel = tilelang.compile(func, out_idx=[-1], target="cuda")

    Args:
        code: Complete Python source code defining a TileLang kernel.
    """
    rejection = _check_code_safety(code)
    if rejection:
        return rejection

    tl, err = _check_tilelang()
    if tl is None:
        return err

    gpu_available, gpu_name = _check_gpu()
    lines = ["## Compile & Benchmark Result", ""]

    # Execute
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102
    except Exception as e:
        lines.append(f"**Compilation failed**: `{type(e).__name__}: {e}`")
        return "\n".join(lines)

    kernel = ns.get("kernel")
    if kernel is None:
        lines.append("**Error**: No `kernel` variable found.")
        return "\n".join(lines)

    lines.append("**Compilation**: Success")
    if gpu_available:
        lines.append(f"**Device**: {gpu_name}")
    lines.append("")

    # Kernel source
    try:
        source = kernel.get_kernel_source()
        if source:
            src_lines = source.strip().split("\n")
            if len(src_lines) > 60:
                source = "\n".join(src_lines[:60]) + f"\n// ... ({len(src_lines) - 60} more lines)"
            lines.extend(["### Generated Kernel Source", "", f"```cuda\n{source}\n```", ""])
    except Exception:
        pass

    # Static analysis
    func = ns.get("func")
    analysis_result = None
    if func is not None:
        try:
            from tilelang.tools.Analyzer import Analyzer
            from tilelang.utils.target import determine_target
            from tilelang.arch import get_arch

            device = get_arch(determine_target("auto"))
            analysis_result = Analyzer.analysis(func, device)

            lines.append("### Static Analysis (Roofline)")
            lines.append("")
            lines.append(f"- **Total FLOPs**: {analysis_result.total_flops:,}")
            lines.append(f"- **Global Memory**: {analysis_result.total_global_bytes:,} bytes "
                         f"({analysis_result.total_global_bytes / 1e6:.2f} MB)")
            lines.append(f"- **Estimated Time**: {analysis_result.estimated_time * 1e3:.4f} ms")
            if analysis_result.expected_tflops:
                lines.append(f"- **Peak TFLOPs**: {analysis_result.expected_tflops:.1f}")
            lines.append(f"- **Peak Bandwidth**: {analysis_result.expected_bandwidth_GBps:.1f} GB/s")
            if analysis_result.total_flops > 0 and analysis_result.total_global_bytes > 0:
                ai = analysis_result.total_flops / analysis_result.total_global_bytes
                lines.append(f"- **Arithmetic Intensity**: {ai:.2f} FLOPs/byte")
            lines.append("")
        except Exception:
            pass

    # GPU benchmark
    if gpu_available:
        try:
            profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Normal)
            latency = profiler.do_bench()
            lines.extend(["### GPU Benchmark", "", f"- **Latency**: {latency:.4f} ms"])

            if analysis_result and analysis_result.total_flops > 0:
                achieved = analysis_result.total_flops / (latency * 1e-3) / 1e12
                lines.append(f"- **Achieved TFLOPs**: {achieved:.2f}")
                if analysis_result.expected_tflops:
                    eff = achieved / analysis_result.expected_tflops * 100
                    lines.append(f"- **Compute Efficiency**: {eff:.1f}%")

            if analysis_result and analysis_result.total_global_bytes > 0:
                bw = analysis_result.total_global_bytes / (latency * 1e-3) / 1e9
                lines.append(f"- **Achieved Bandwidth**: {bw:.1f} GB/s")
                if analysis_result.expected_bandwidth_GBps > 0:
                    bw_eff = bw / analysis_result.expected_bandwidth_GBps * 100
                    lines.append(f"- **Bandwidth Efficiency**: {bw_eff:.1f}%")

            lines.append("")
        except Exception as e:
            lines.append(f"**Benchmark failed**: `{type(e).__name__}: {e}`")
            lines.append("")
    else:
        lines.extend(["**Note**: No GPU detected. Skipping benchmark.",
                       "Compilation succeeded — deploy to a GPU machine for performance numbers.", ""])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Tool 2: analyze_kernel
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def analyze_kernel(code: str) -> str:
    """Analyze a TileLang kernel using the roofline performance model.

    Uses tilelang's Analyzer to statically analyze the kernel IR and compute:
    - Total FLOPs (from ``T.gemm`` operations)
    - Total global memory bytes (from ``T.copy`` operations)
    - Estimated execution time (roofline model)
    - Whether the kernel is compute-bound or memory-bound
    - Arithmetic intensity

    The code must define a ``@T.prim_func`` function assigned to ``func``.
    If ``kernel`` is also present, real GPU benchmarking is included.

    Args:
        code: Python source defining a TileLang PrimFunc assigned to ``func``.
    """
    tl, err = _check_tilelang()
    if tl is None:
        return err

    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102
    except Exception as e:
        return f"**Error executing code**: `{type(e).__name__}: {e}`"

    func = ns.get("func")
    if func is None:
        return ("**Error**: No `func` variable found. "
                "Assign your `@T.prim_func` to a variable named `func`.")

    try:
        from tilelang.tools.Analyzer import Analyzer
        from tilelang.utils.target import determine_target
        from tilelang.arch import get_arch

        target_str = determine_target("auto")
        device = get_arch(target_str)
        r = Analyzer.analysis(func, device)
    except Exception as e:
        return f"**Analysis failed**: `{type(e).__name__}: {e}`"

    cc = getattr(device, "compute_capability", "unknown")
    lines = ["## Kernel Analysis (Roofline Model)", "",
             f"**Device**: compute capability {cc}", "",
             "### Compute", "",
             f"- **Total FLOPs**: {r.total_flops:,}"]
    if r.expected_tflops:
        lines.append(f"- **Peak TFLOPs**: {r.expected_tflops:.1f}")
    lines.extend(["", "### Memory", "",
                   f"- **Global Memory Bytes**: {r.total_global_bytes:,} "
                   f"({r.total_global_bytes / 1e6:.2f} MB)",
                   f"- **Peak Bandwidth**: {r.expected_bandwidth_GBps:.1f} GB/s",
                   "", "### Roofline Estimate", "",
                   f"- **Estimated Time**: {r.estimated_time * 1e3:.4f} ms"])

    if r.total_flops > 0 and r.expected_tflops:
        compute_time = r.total_flops / (r.expected_tflops * 1e12)
        mem_time = r.total_global_bytes / (r.expected_bandwidth_GBps * 1e9)
        ai = r.total_flops / r.total_global_bytes if r.total_global_bytes > 0 else float("inf")
        lines.append(f"- **Arithmetic Intensity**: {ai:.2f} FLOPs/byte")
        lines.append(f"- **Compute Time**: {compute_time * 1e3:.4f} ms")
        lines.append(f"- **Memory Time**: {mem_time * 1e3:.4f} ms")
        lines.append(f"- **Bottleneck**: {'Compute-bound' if compute_time > mem_time else 'Memory-bound'}")
    lines.append("")

    # Real benchmark if kernel present
    kernel = ns.get("kernel")
    if kernel is not None:
        gpu_ok, _ = _check_gpu()
        if gpu_ok:
            try:
                profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Normal)
                latency = profiler.do_bench()
                lines.extend(["### Actual GPU Benchmark", "",
                               f"- **Measured Latency**: {latency:.4f} ms"])
                ratio = (r.estimated_time * 1e3) / latency if latency > 0 else 0
                lines.append(f"- **Roofline / Actual**: {ratio:.2f}x")
                if r.total_flops > 0 and r.expected_tflops:
                    achieved = r.total_flops / (latency * 1e-3) / 1e12
                    eff = achieved / r.expected_tflops * 100
                    lines.append(f"- **Achieved TFLOPs**: {achieved:.2f} ({eff:.1f}% of peak)")
                lines.append("")
            except Exception:
                pass

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Tool 3: autotune
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def autotune(code: str, configs: str = "") -> str:
    """Run tilelang's AutoTuner to find the best kernel configuration.

    Wraps ``AutoTuner.run()`` — parallel compilation, SHA256 result caching,
    timeout handling, and optional correctness validation.

    The code must define:
    - ``kernel_fn``: a function that takes config keyword arguments and returns
      a ``@T.prim_func``.
    - ``configs``: a list of config dicts to search over.

    Example::

        import tilelang.language as T

        M, N, K = 4096, 4096, 4096

        def kernel_fn(block_M=128, block_N=128, block_K=32, num_stages=3, threads=128):
            @T.prim_func
            def func(A: T.Tensor((M, K), T.float16),
                     B: T.Tensor((K, N), T.float16),
                     C: T.Tensor((M, N), T.float16)):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                    A_s = T.alloc_shared((block_M, block_K), T.float16)
                    B_s = T.alloc_shared((block_K, block_N), T.float16)
                    C_f = T.alloc_fragment((block_M, block_N), T.float32)
                    T.clear(C_f)
                    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[by * block_M, ko * block_K], A_s)
                        T.copy(B[ko * block_K, bx * block_N], B_s)
                        T.gemm(A_s, B_s, C_f)
                    T.copy(C_f, C[by * block_M, bx * block_N])
            return func

        configs = [
            dict(block_M=m, block_N=n, block_K=k, num_stages=s, threads=t)
            for m in [64, 128] for n in [64, 128]
            for k in [32, 64] for s in [2, 3] for t in [128, 256]
        ]

    Args:
        code: Python source defining ``kernel_fn`` and ``configs``.
        configs: JSON string of config list (alternative to defining in code).
    """
    tl, err = _check_tilelang()
    if tl is None:
        return err

    gpu_ok, gpu_name = _check_gpu()
    if not gpu_ok:
        return "**Error**: No GPU detected. AutoTuner requires a CUDA/ROCm GPU."

    rejection = _check_code_safety(code)
    if rejection:
        return rejection

    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102
    except Exception as e:
        return f"**Error executing code**: `{type(e).__name__}: {e}`"

    kernel_fn = ns.get("kernel_fn")
    if kernel_fn is None:
        return ("**Error**: No `kernel_fn` variable found. "
                "Define a function that takes config kwargs and returns a `@T.prim_func`.")

    config_list = ns.get("configs")
    if config_list is None and configs:
        import json
        try:
            config_list = json.loads(configs)
        except json.JSONDecodeError as e:
            return f"**Error**: Invalid configs JSON: {e}"

    if not config_list:
        return "**Error**: No configs provided. Define `configs` as a list of dicts."

    lines = ["## AutoTuner Results", "",
             f"**Device**: {gpu_name}",
             f"**Configs**: {len(config_list)}",
             ""]

    try:
        from tilelang.autotuner import AutoTuner

        result = (
            AutoTuner.from_kernel(kernel=kernel_fn, configs=config_list)
            .set_compile_args(out_idx=[-1], target="auto")
            .set_profile_args(skip_check=True)
            .run()
        )

        lines.append("### Best Configuration")
        lines.append("")
        if result.config:
            for k, v in result.config.items():
                lines.append(f"- `{k} = {v}`")
        lines.append("")

        if result.latency is not None:
            lines.append(f"**Best Latency**: {result.latency:.4f} ms")

        if result.ref_latency is not None:
            lines.append(f"**Reference Latency**: {result.ref_latency:.4f} ms")
            if result.latency and result.ref_latency:
                speedup = result.ref_latency / result.latency
                lines.append(f"**Speedup**: {speedup:.2f}x")

        lines.append("")

        # Show generated source if available
        if result.libcode:
            src_lines = result.libcode.strip().split("\n")
            preview = result.libcode
            if len(src_lines) > 60:
                preview = "\n".join(src_lines[:60]) + f"\n// ... ({len(src_lines) - 60} more lines)"
            lines.extend(["### Best Kernel Source", "",
                           f"```cuda\n{preview}\n```", ""])

    except Exception as e:
        lines.append(f"**AutoTuner failed**: `{type(e).__name__}: {e}`")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Tool 4: search_examples
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def search_examples(query: str, max_results: int = 10) -> str:
    """Search TileLang example files by keyword.

    Scans example filenames and file contents for matching keywords.

    Args:
        query: Search keyword (e.g. "flash_attention", "gemm", "convolution").
        max_results: Maximum number of results to return (default 10).
    """
    root = _find_tilelang_root()
    if not root:
        return "TileLang source tree not found. Cannot search examples."

    examples_dir = os.path.join(root, "examples")
    if not os.path.isdir(examples_dir):
        return f"Examples directory not found at {examples_dir}"

    query_lower = query.lower()
    results: list[dict[str, str]] = []

    for dirpath, _dirnames, filenames in os.walk(examples_dir):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(filepath, root)

            if query_lower in fname.lower() or query_lower in rel_path.lower():
                results.append({"path": rel_path, "match": "filename"})
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.readlines()[:200]
                for i, line in enumerate(head):
                    if query_lower in line.lower():
                        results.append({
                            "path": rel_path,
                            "match": f"line {i + 1}: {line.strip()[:100]}",
                        })
                        break
            except OSError:
                continue

            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    if not results:
        return f"No examples found matching '{query}'."

    lines = [f"## Examples matching '{query}'", ""]
    for r in results:
        lines.append(f"- `{r['path']}` ({r['match']})")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Tool 5: read_example
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def read_example(path: str) -> str:
    """Read the contents of a TileLang example file.

    Args:
        path: Relative path from tilelang root, e.g. "examples/gemm/example_gemm.py".
    """
    root = _find_tilelang_root()
    if not root:
        return "TileLang source tree not found."

    filepath = os.path.join(root, path)
    real_root = os.path.realpath(root)
    real_file = os.path.realpath(filepath)
    if not real_file.startswith(real_root):
        return "Invalid path: must be within the tilelang directory."

    if not os.path.isfile(real_file):
        return f"File not found: {path}"

    try:
        with open(real_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        lines = content.split("\n")
        if len(lines) > 500:
            content = "\n".join(lines[:500]) + f"\n\n... ({len(lines) - 500} more lines truncated)"
        return f"```python\n{content}\n```"
    except OSError as e:
        return f"Error reading file: {e}"
