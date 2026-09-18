#!/usr/bin/env python3
"""
Batch correctness runner for KernelBench on MetaX C500 (MACA).

For each runnable problem, evaluates a hand-written CUDA kernel (the reference-level
kernel that ships with KernelBench examples where available, otherwise a generated
kernel via the problem's own Model as ModelNew passthrough is NOT used — instead we
synthesize a simple custom kernel per problem using the dataset's Model as the spec).

Phase-1 scope: build + correctness only (no performance), so we evaluate the
'reference implementation compiled through the harness' path: each problem's
ref arch is executed via the harness to confirm the toolchain works end-to-end
for that problem's shapes and operators. This is the correctness loop closure.

Failure taxonomy (per c500-a2a-adapter):
  backend_error / algorithm_error / invalid_infrastructure
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent / "KernelBench"
sys.path.insert(0, str(REPO / "src"))

from kernelbench import eval as kernel_eval  # noqa: E402
from kernelbench import dataset as kb_dataset  # noqa: E402


def classify_error(err: Exception, metadata: dict) -> str:
    """Classify a failure into the migration taxonomy."""
    msg = str(err).lower()
    meta_str = json.dumps({k: str(v) for k, v in metadata.items()}, default=str).lower()
    combined = msg + " " + meta_str

    oom_markers = [
        "out of memory", "outofmemory", "cuda error: out of memory",
        "alloc", "memory", "no space left", "exceed",
    ]
    if any(m in combined for m in oom_markers):
        return "invalid_infrastructure"

    backend_markers = [
        "mxcc", "cucc", "cu-bridge", "maca", "compile", "nvcc", "link",
        "undefined symbol", "unresolved", "kernel image", "no kernel image",
        "launch failure", "illegal memory", "segfault", "device side assert",
    ]
    if any(m in combined for m in backend_markers):
        return "backend_error"

    return "algorithm_error"


def check_gpu_memory_feasible(approx_elems: int) -> bool:
    """Rough pre-filter: input + ref out + new out must fit in GPU memory."""
    total_bytes = approx_elems * 4 * 3  # fp32, 3 copies
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return total_bytes < free * 0.8
    return total_bytes < 13 * 1e9


def static_input_elems(code: str) -> int:
    """Best-effort static input element count (returns 0 if unresolved)."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    env = {}

    def setval(targets, v):
        if isinstance(targets, ast.Tuple) and isinstance(v, tuple):
            for t, vv in zip(targets.elts, v):
                if isinstance(t, ast.Name):
                    env[t.id] = vv
        elif isinstance(targets, ast.Name):
            env[targets.id] = v

    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                v = ast.literal_eval(node.value)
                for t in node.targets:
                    setval(t, v)
            except Exception:
                pass
    for _ in range(3):
        for node in tree.body:
            if isinstance(node, ast.Assign):
                try:
                    v = eval(compile(ast.Expression(body=node.value), "<s>", "eval"), {}, dict(env))
                    for t in node.targets:
                        setval(t, v)
                except Exception:
                    pass

    elems = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("rand", "randn", "zeros", "ones", "empty"):
                dims = []
                for a in node.args:
                    v = None
                    if isinstance(a, ast.Name):
                        v = env.get(a.id)
                    if v is None:
                        try:
                            v = eval(compile(ast.Expression(body=a), "<s>", "eval"), {}, dict(env))
                        except Exception:
                            try:
                                v = ast.literal_eval(a)
                            except Exception:
                                v = None
                    dims.append(v)
                if all(isinstance(x, int) for x in dims) and dims:
                    m = 1
                    for x in dims:
                        m *= x
                    elems += m
    return elems


def build_modelnew_from_ref(ref_code: str) -> str:
    """
    Build a ModelNew implementation that calls into a custom CUDA kernel,
    by reusing the reference Model class and wrapping it with a simple
    in-place op chain. For Phase 1 (build + correctness loop), we verify the
    toolchain end-to-end by compiling the reference model through load_inline:
    we take the ref Model source and add a trivial custom CUDA kernel wrapper.
    """
    # This is a placeholder - real per-problem kernels are handled by the
    # kernel generator. For phase-1 loop verification we use the reference
    # itself compiled through the CUDA toolchain path.
    return None


def run_one(level: int, problem_id: int, dataset, verbose: bool = False, build_root: str = None) -> dict:
    """Run correctness evaluation for one problem. Returns result dict."""
    t0 = time.time()
    result = {
        "level": level,
        "problem_id": problem_id,
        "status": "unknown",
        "compiled": False,
        "correctness": False,
        "trials": "0/0",
        "failure_class": "unknown",
        "error": None,
        "wall_seconds": 0,
    }

    try:
        problem = dataset.get_problem_by_id(problem_id)
        result["name"] = problem.name
        ref_code = problem.code

        # static memory pre-filter
        elems = static_input_elems(ref_code)
        result["static_input_elems"] = elems
        if elems > 0 and not check_gpu_memory_feasible(elems):
            result["status"] = "skipped"
            result["failure_class"] = "invalid_infrastructure"
            result["error"] = f"static input too large for C500 16.3GB: {elems} elems ({elems*4/1e9:.2f} GB)"
            result["wall_seconds"] = time.time() - t0
            return result

        # Phase-1: verify the harness can load, build, and run this problem's
        # reference through the full pipeline. We construct a ModelNew that
        # is identical to the ref Model (a no-op custom kernel) to exercise
        # compile + correctness without needing LLM-generated kernels.
        # The simplest correct ModelNew: copy of Model with a trivial inline
        # CUDA kernel applied to the output.
        model_new_code = build_passthrough_modelnew(ref_code, problem_id)

        build_dir = None
        if build_root:
            build_dir = os.path.join(build_root, f"l{level}_p{problem_id}")

        device = torch.device("cuda:0")

        eval_result = kernel_eval.eval_kernel_against_ref(
            original_model_src=ref_code,
            custom_model_src=model_new_code,
            seed_num=42,
            num_correct_trials=1,
            num_perf_trials=0,
            measure_performance=False,
            verbose=verbose,
            build_dir=build_dir,
            device=device,
            backend="cuda",
            precision=torch.float32,
        )

        if eval_result is None:
            result["status"] = "failed"
            result["failure_class"] = "backend_error"
            result["error"] = "eval returned None (lock/retry condition)"
        else:
            result["compiled"] = eval_result.compiled
            result["correctness"] = eval_result.correctness
            result["trials"] = eval_result.metadata.get("correctness_trials", "0/0")
            result["metadata"] = {
                k: str(v)[:500] for k, v in eval_result.metadata.items()
                if k in ("compilation_error_name", "runtime_error_name", "max_difference",
                         "avg_difference", "correctness_issue", "hardware", "device",
                         "correctness_trials")
            }
            if eval_result.compiled and eval_result.correctness:
                result["status"] = "passed"
                result["failure_class"] = "none"
            elif eval_result.compiled:
                result["status"] = "failed"
                result["failure_class"] = "algorithm_error"
                result["error"] = "correctness mismatch"
            else:
                result["status"] = "failed"
                result["failure_class"] = "backend_error"
                err = eval_result.metadata.get("compilation_error", None)
                result["error"] = str(err)[:400] if err else "compilation failure"

    except Exception as e:
        result["status"] = "failed"
        result["failure_class"] = classify_error(e, {})
        result["error"] = f"{type(e).__name__}: {str(e)[:400]}"

    result["wall_seconds"] = round(time.time() - t0, 2)
    return result


def build_passthrough_modelnew(ref_code: str, problem_id: int) -> str:
    """
    Construct a ModelNew that runs the reference model's forward and adds a
    trivial custom CUDA kernel (elementwise identity) on the output, so the
    full compile/load/run path is exercised on real CUDA code.
    """
    # Extract the Model class body and build a subclass that calls a custom kernel
    return '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void identity_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 0.0f;
    }
}

torch::Tensor identity_op(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    identity_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n
    );
    return out;
}
"""

cpp_source = "torch::Tensor identity_op(torch::Tensor x);"

_identity_module = load_inline(
    name="c500_identity_kernel",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=["identity_op"],
    verbose=False,
)

''' + _wrap_model_with_identity(ref_code)


def _wrap_model_with_identity(ref_code: str) -> str:
    """
    Take the ref code which defines `Model`, and emit a module that also defines
    `ModelNew` with the same forward, applying the identity kernel to the output.
    """
    # The ref code already defines Model, get_inputs, get_init_inputs.
    # We append a ModelNew that wraps it.
    wrapper = '''

class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Model(*args, **kwargs)
    def forward(self, *x):
        out = self.model(*x)
        if isinstance(out, torch.Tensor):
            return _identity_module.identity_op(out)
        if isinstance(out, (list, tuple)):
            return [_identity_module.identity_op(t) if isinstance(t, torch.Tensor) else t for t in out]
        return out
'''
    return ref_code + wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=str, default="1,2,3")
    ap.add_argument("--problem-ids", type=str, default="",
                    help="comma list of ids, empty = all")
    ap.add_argument("--out", type=str,
                    default="/data/cuda-harness-migration/batch-results.jsonl")
    ap.add_argument("--build-root", type=str,
                    default="/data/cuda-harness-migration/batch_build")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit problems per level (0=all)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    levels = [int(l) for l in args.levels.split(",")]
    problem_ids = [int(p) for p in args.problem_ids.split(",")] if args.problem_ids else None

    results = []
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"[batch] GPU: {torch.cuda.get_device_name(device)}")

    for level in levels:
        ds = kb_dataset.construct_kernelbench_dataset(level=level, source="local")
        ids = problem_ids or ds.get_problem_ids()
        if args.limit:
            ids = ids[: args.limit]
        print(f"[batch] level {level}: {len(ids)} problems")

        for pid in ids:
            result = run_one(level, pid, ds, verbose=args.verbose,
                             build_root=args.build_root)
            results.append(result)
            # append incrementally so partial results survive crashes
            with open(out_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
            torch.cuda.empty_cache()
            status = result["status"]
            emoji = {"passed": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(status, "?")
            err = f" [{result['failure_class']}] {result.get('error', '')[:100]}" if status != "passed" else ""
            print(f"  L{level} P{pid:>3} {result.get('name', '?')[:60]:<60} {emoji} {status}{err}")

    # summary
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n[summary] passed={passed} failed={failed} skipped={skipped} total={len(results)}")
    summary_path = out_path.parent / (out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "passed": passed, "failed": failed, "skipped": skipped, "total": len(results),
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))
    print(f"[summary] written to {summary_path}")


if __name__ == "__main__":
    main()
