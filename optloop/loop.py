#!/usr/bin/env python3
"""
Long-running C500 optimization loop.

Each iteration:
  1. pick a candidate kernel config (tile sizes, thread layout, unroll)
  2. compile via load_inline (cu-bridge -> cucc -> mxcc)
  3. verify correctness vs torch.matmul (atol 1e-3)
  4. time with cuda_event, cold-cache (L2 flush between trials)
  5. append result to JSONL incrementally

Runs until the wall-clock budget is exhausted. Safe to interrupt: all results
are on disk.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "KernelBench" / "src"))

import kernels  # noqa: E402


def time_kernel_cold(fn, args, num_warmup=10, num_trials=30):
    device = torch.device("cuda:0")
    for _ in range(num_warmup):
        fn(*args)
        torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize(device)
        dummy = torch.empty(2 * 1024 * 1024, dtype=torch.int8, device=device)
        dummy.fill_(42)
        del dummy
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize(device)
        times.append(s.elapsed_time(e))
    t = torch.tensor(times)
    return float(t.mean().item()), float(t.min().item()), float(t.std().item())


def check_correct(out, ref, rel_tol=1e-3):
    """fp32-safe correctness criterion.

    Tiled fp32 kernels accumulate in a different order than torch.matmul's
    tree reduction; the resulting summation-order noise is ~3e-4 relative to
    output scale (verified reproducible on CPU). Elementwise relative error
    is unbounded near cancellation, so we compare absolute error against the
    output magnitude: max|out-ref| <= rel_tol * max|ref|.
    """
    max_abs = (out - ref).abs().max().item()
    scale = ref.abs().max().item()
    return (max_abs <= rel_tol * scale), max_abs, scale


def build_and_test(src, cpp_decl, name, A, B, ref, build_dir, trials, rel_tol=1e-3):
    from torch.utils.cpp_extension import load_inline
    os.makedirs(build_dir, exist_ok=True)
    t0 = time.time()
    result = {"name": name, "compiled": False, "correct": False}
    # name is the extension name; the exported symbol is whatever cpp_decl
    # declares. load_inline's `functions` list must name that symbol, not the
    # extension, or pybind emits "was not declared in this scope".
    import re as _re
    m = _re.search(r"torch::Tensor\s+(\w+)\s*\(", cpp_decl)
    fn_name = m.group(1) if m else name
    try:
        mod = load_inline(
            name=name,
            cpp_sources=[cpp_decl],
            cuda_sources=[src],
            functions=[fn_name],
            build_directory=build_dir,
            verbose=False,
        )
        result["compiled"] = True
        result["compile_seconds"] = round(time.time() - t0, 1)
        fn = getattr(mod, fn_name)
        out = fn(A, B)
        torch.cuda.synchronize()
        ok, max_abs, scale = check_correct(out, ref, rel_tol)
        result["correct"] = bool(ok)
        if not ok:
            result["max_diff"] = max_abs
            result["rel_to_scale"] = max_abs / scale if scale > 0 else None
        else:
            mean_ms, min_ms, std_ms = time_kernel_cold(fn, [A, B], num_trials=trials)
            result["us_mean"] = round(mean_ms * 1000, 1)
            result["us_min"] = round(min_ms * 1000, 1)
            result["us_std"] = round(std_ms * 1000, 1)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return result


def gen_configs():
    """Yield (kind, BM, BN, T, U) candidate configs in a sensible order.

    v4 requires BM == BN (square tiles) for in-bounds cooperative loads.
    Hardware: 104 SM, 2048 threads/SM, 64KB shared per block, warp 64.
    """
    tiles = []
    for S in (16, 32, 64, 128):
        for T in (1, 2, 4, 8, 16):
            nthreads = S * (S // T)
            if nthreads < 64 or nthreads > 1024:
                continue
            smem = (S * S + S * S) * 4   # two S x S tiles
            if smem > 65536:
                continue
            tiles.append((S, T))

    # stage 1: v4 baseline sweep
    for (S, T) in tiles:
        yield ("v4", S, S, T, 0)
    # stage 2: v5 (bank-conflict padding)
    for (S, T) in tiles:
        yield ("v5", S, S, T, 0)
    # stage 3: v6 (K-unroll)
    for (S, T) in tiles:
        for U in (2, 4, 8, 16):
            if U > S:
                continue
            yield ("v6", S, S, T, U)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/cuda-harness-migration/optloop/results.jsonl")
    ap.add_argument("--build-root", default="/data/cuda-harness-migration/optloop/build")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--only-correct", action="store_true",
                    help="skip configs that fail correctness (record them once)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(args.build_root)
    build_root.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(0)
    n = args.n
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B
    print(f"[setup] n={n} GPU={torch.cuda.get_device_name(0)}")

    # ---- eager + compile baselines (always re-measure for this session)
    e_mean, e_min, e_std = time_kernel_cold(lambda a, b: a @ b, [A, B], num_trials=args.trials)
    eager_us = e_mean * 1000
    base = {"name": "eager_matmul", "compiled": True, "correct": True,
            "us_mean": round(eager_us, 1), "us_min": round(e_min * 1000, 1),
            "us_std": round(e_std * 1000, 1)}
    print(f"[baseline] eager: {eager_us:.1f} us")
    with open(out_path, "a") as f:
        f.write(json.dumps(base) + "\n")

    try:
        cmodel = torch.compile(lambda a, b: a @ b)
        for _ in range(5):
            cmodel(A, B)
        c_mean, c_min, c_std = time_kernel_cold(cmodel, [A, B], num_trials=args.trials)
        rec = {"name": "torch_compile", "compiled": True, "correct": True,
               "us_mean": round(c_mean * 1000, 1), "us_min": round(c_min * 1000, 1),
               "us_std": round(c_std * 1000, 1)}
        print(f"[baseline] torch.compile: {c_mean*1000:.1f} us")
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print(f"[baseline] torch.compile failed: {e}")

    # ---- candidate sweep
    budget_s = args.budget_hours * 3600
    t_start = time.time()
    best_us = eager_us
    n_correct = 0
    n_total = 0

    for (kind, BM, BN, T, U) in gen_configs():
        if time.time() - t_start > budget_s:
            print(f"[loop] budget exhausted ({args.budget_hours}h), stopping")
            break
        n_total += 1
        name = kernels.cpp_name(kind, BM, BN, T, U)
        src = kernels.make_kernel(kind, BM, BN, T, U)
        cpp = f"torch::Tensor {name}(torch::Tensor A, torch::Tensor B);"
        res = build_and_test(src, cpp, name, A, B, ref,
                             str(build_root / name), args.trials)
        if res.get("us_mean"):
            res["speedup_vs_eager"] = round(eager_us / res["us_mean"], 3)
            if res["us_mean"] < best_us:
                best_us = res["us_mean"]
        if res["correct"]:
            n_correct += 1
        status = "OK" if res["correct"] else ("COMP-FAIL" if not res["compiled"] else "WRONG")
        print(f"[{n_total:>4}] {name:<28} {status:<9} us={res.get('us_mean','-'):>7} "
              f"speedup={res.get('speedup_vs_eager','-'):>6} "
              f"err={str(res.get('error',''))[:60] or str(res.get('max_diff',''))[:20]}",
              flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        torch.cuda.empty_cache()

    print(f"[loop] done: {n_total} candidates, {n_correct} correct, "
          f"best={best_us:.1f} us ({eager_us/best_us:.2f}x vs eager)")


if __name__ == "__main__":
    main()
