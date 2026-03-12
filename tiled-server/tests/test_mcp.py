"""Tests for tiled MCP server tools and resources."""

import pytest

from tiled_server.mcp import (
    compile_and_benchmark,
    analyze_kernel,
    autotune,
    search_examples,
    read_example,
    list_apis,
    lookup_api as lookup_api_tool,
    get_templates,
    get_optimization_tips,
    # Resources
    api_index,
    api_lookup,
    templates_index,
    optimization_tips_resource,
    best_practices_resource,
)


# ═══════════════════════════════════════════════════════════════════
#  Resources
# ═══════════════════════════════════════════════════════════════════


class TestApiIndex:
    def test_returns_markdown(self):
        result = api_index()
        assert "# TileLang API Reference" in result

    def test_has_categories(self):
        result = api_index()
        assert "memory" in result.lower()
        assert "compute" in result.lower()
        assert "loop" in result.lower()

    def test_has_symbols(self):
        result = api_index()
        assert "alloc_shared" in result
        assert "gemm" in result


class TestApiLookup:
    def test_known_symbol(self):
        result = api_lookup("gemm")
        assert "gemm" in result.lower()

    def test_with_prefix(self):
        result = api_lookup("T.alloc_shared")
        assert "alloc_shared" in result

    def test_tilelang_prefix(self):
        result = api_lookup("tilelang.jit")
        assert "jit" in result.lower()

    def test_unknown_symbol(self):
        result = api_lookup("nonexistent_symbol_xyz")
        assert "not found" in result.lower()

    def test_dtype(self):
        result = api_lookup("float16")
        assert "float16" in result


class TestTemplatesIndex:
    def test_returns_all_templates(self):
        result = templates_index()
        assert "# TileLang Kernel Templates" in result
        assert "gemm" in result.lower()
        assert "elementwise" in result.lower()
        assert "flash_attention" in result.lower()

    def test_has_code(self):
        result = templates_index()
        assert "```python" in result
        assert "T.gemm" in result


class TestOptimizationTips:
    def test_returns_tips(self):
        result = optimization_tips_resource()
        assert "# TileLang Optimization Tips" in result
        assert "pipeline" in result.lower() or "Pipelined" in result

    def test_has_example_code(self):
        result = optimization_tips_resource()
        assert "```python" in result

    def test_has_multiple_topics(self):
        result = optimization_tips_resource()
        # Should have several ## sections
        assert result.count("## ") >= 3


class TestBestPractices:
    def test_returns_practices(self):
        result = best_practices_resource()
        assert "# TileLang Best Practices" in result

    def test_has_categories(self):
        result = best_practices_resource()
        assert "Memory Hierarchy" in result
        assert "Compute" in result

    def test_not_empty(self):
        result = best_practices_resource()
        assert len(result) > 100


# ═══════════════════════════════════════════════════════════════════
#  Knowledge query tools (list_apis, lookup_api, get_templates, get_optimization_tips)
# ═══════════════════════════════════════════════════════════════════


class TestListApis:
    def test_all_categories(self):
        result = list_apis()
        assert "memory" in result.lower()
        assert "compute" in result.lower()
        assert "alloc_shared" in result

    def test_filter_by_category(self):
        result = list_apis(category="memory")
        assert "alloc_shared" in result
        assert "gemm" not in result

    def test_unknown_category(self):
        result = list_apis(category="nonexistent_xyz")
        assert "Unknown category" in result


class TestLookupApiTool:
    def test_known_symbol(self):
        result = lookup_api_tool("gemm")
        assert "gemm" in result.lower()

    def test_with_prefix(self):
        result = lookup_api_tool("T.alloc_shared")
        assert "alloc_shared" in result

    def test_unknown(self):
        result = lookup_api_tool("nonexistent_xyz")
        assert "not found" in result.lower()


class TestGetTemplates:
    def test_returns_templates(self):
        result = get_templates()
        assert "gemm" in result.lower()
        assert "```python" in result

    def test_has_all_templates(self):
        result = get_templates()
        assert "elementwise" in result.lower()
        assert "flash_attention" in result.lower()


class TestGetOptimizationTips:
    def test_returns_tips_and_practices(self):
        result = get_optimization_tips()
        assert "Optimization Tips" in result
        assert "Best Practices" in result

    def test_has_code_examples(self):
        result = get_optimization_tips()
        assert "```python" in result


# ═══════════════════════════════════════════════════════════════════
#  Tool: compile_and_benchmark
# ═══════════════════════════════════════════════════════════════════


class TestCompileAndBenchmark:
    def test_rejects_dangerous_code(self):
        result = compile_and_benchmark("import os; os.system('rm -rf /')")
        assert "Rejected" in result

    def test_rejects_subprocess(self):
        result = compile_and_benchmark("import subprocess; subprocess.run(['ls'])")
        assert "Rejected" in result

    def test_invalid_syntax(self):
        result = compile_and_benchmark("def foo(:\n  pass")
        assert "failed" in result.lower() or "not installed" in result.lower()

    def test_no_kernel_variable(self):
        result = compile_and_benchmark("x = 42")
        assert "No `kernel` variable" in result or "not installed" in result.lower()

    def test_tilelang_not_available(self):
        result = compile_and_benchmark("kernel = None")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════
#  Tool: analyze_kernel
# ═══════════════════════════════════════════════════════════════════


class TestAnalyzeKernel:
    def test_no_func_variable(self):
        result = analyze_kernel("x = 42")
        assert "No `func` variable" in result or "not installed" in result.lower()

    def test_syntax_error(self):
        result = analyze_kernel("def foo(:\n  pass")
        assert "Error" in result or "not installed" in result.lower()

    def test_returns_string(self):
        result = analyze_kernel("func = None")
        assert isinstance(result, str)

    def test_tilelang_not_installed_or_analysis(self):
        code = (
            "import tilelang.language as T\n"
            "M, N, K = 1024, 1024, 1024\n"
            "BM, BN, BK = 128, 128, 32\n"
            "@T.prim_func\n"
            "def func(A: T.Tensor((M, K), T.float16),\n"
            "         B: T.Tensor((K, N), T.float16),\n"
            "         C: T.Tensor((M, N), T.float16)):\n"
            "    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):\n"
            "        A_s = T.alloc_shared((BM, BK), T.float16)\n"
            "        B_s = T.alloc_shared((BK, BN), T.float16)\n"
            "        C_f = T.alloc_fragment((BM, BN), T.float32)\n"
            "        T.clear(C_f)\n"
            "        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):\n"
            "            T.copy(A[by * BM, ko * BK], A_s)\n"
            "            T.copy(B[ko * BK, bx * BN], B_s)\n"
            "            T.gemm(A_s, B_s, C_f)\n"
            "        T.copy(C_f, C[by * BM, bx * BN])\n"
        )
        result = analyze_kernel(code)
        assert isinstance(result, str)
        if "not installed" not in result.lower():
            assert "FLOPs" in result
            assert "Roofline" in result


# ═══════════════════════════════════════════════════════════════════
#  Tool: autotune
# ═══════════════════════════════════════════════════════════════════


class TestAutotune:
    def test_no_kernel_fn(self):
        result = autotune("x = 42")
        assert ("No `kernel_fn`" in result
                or "not installed" in result.lower()
                or "No GPU" in result)

    def test_no_configs(self):
        result = autotune("def kernel_fn(): pass")
        assert ("No configs" in result
                or "not installed" in result.lower()
                or "No GPU" in result)

    def test_rejects_dangerous_code(self):
        result = autotune("import os; os.system('rm -rf /')")
        assert "Rejected" in result or "not installed" in result.lower() or "No GPU" in result

    def test_returns_string(self):
        result = autotune("kernel_fn = lambda: None\nconfigs = [{}]")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════
#  Tool: search_examples
# ═══════════════════════════════════════════════════════════════════


class TestSearchExamples:
    def test_search_gemm(self):
        result = search_examples("gemm")
        assert isinstance(result, str)
        if "not found" not in result.lower():
            assert "gemm" in result.lower()

    def test_search_no_results(self):
        result = search_examples("zzz_nonexistent_pattern_zzz")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════
#  Tool: read_example
# ═══════════════════════════════════════════════════════════════════


class TestReadExample:
    def test_invalid_path_traversal(self):
        result = read_example("../../etc/passwd")
        assert "Invalid path" in result or "not found" in result.lower()

    def test_nonexistent_file(self):
        result = read_example("examples/nonexistent_file.py")
        assert "not found" in result.lower()
