#!/usr/bin/env python3
"""
Long-running C500 kernel optimization loop (phase 2).

Runs an extended parameter sweep over the wmma3 family (shared-staged fp16
tensor-core matmul) plus refinements, until a wall-clock budget is exhausted.
All results are appended to a JSONL file incrementally so partial progress
survives interruption.

Stages:
  1. baseline (eager, torch.compile)
  2. wmma3 tile sweep (BM, BN, BK, WM, WN)
  3. best-config refinement: K-unrolling, double buffering, block-stride
  4. (stretch) wmma3 with L2 cache hints / persistent kernels
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

import kernels_wmma3  # noqa: E402
from loop import build_and_test, time_kernel_cold  # noqa: E402


def wmma3_configs():
    """Generate valid wmma3 configs in performance-relevant order.

    Constraints:
      - BK must be a multiple of 16 (wmma K fragment size)
      - shared (BM*(BK+8) + BK*(BN+8)) * 2 bytes <= 64KB
      - threads 64..1024, and BM % (16*WM) == 0, BN % (16*WN) == 0
    """
    configs = []
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            for BK in (16, 32, 64):
                if BK % 16 != 0:
                    continue
                smem = (BM * (BK + 8) + BK * (BN + 8)) * 2  # fp16 = 2 bytes
                if smem > 65536:
                    continue
                for WM in (1, 2, 4, 8):
                    for WN in (1, 2, 4, 8):
                        if BM % (16 * WM) or BN % (16 * WN):
                            continue
                        nwarps = (BM // (16 * WM)) * (BN // (16 * WN))
                        nthreads = 64 * nwarps
                        if nthreads < 64 or nthreads > 1024:
                            continue
                        configs.append((BM, BN, BK, WM, WN))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/cuda-harness-migration/optloop/longloop-results.jsonl")
    ap.add_argument("--build-root", default="/data/cuda-harness-migration/optloop/longbuild")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--rel-tol", type=float, default=5e-3)
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

    # baselines
    e_mean, e_min, e_std = time_kernel_cold(lambda a, b: a @ b, [A, B], num_trials=args.trials)
    eager_us = e_mean * 1000
    print(f"[baseline] eager: {eager_us:.1f} us")
    with open(out_path, "a") as f:
        f.write(json.dumps({"name": "eager_matmul", "correct": True,
                            "us_mean": round(eager_us, 1),
                            "us_min": round(e_min * 1000, 1)}) + "\n")

    try:
        cmodel = torch.compile(lambda a, b: a @ b)
        for _ in range(5):
            cmodel(A, B)
        c_mean, c_min, _ = time_kernel_cold(cmodel, [A, B], num_trials=args.trials)
        print(f"[baseline] torch.compile: {c_mean*1000:.1f} us")
        with open(out_path, "a") as f:
            f.write(json.dumps({"name": "torch_compile", "correct": True,
                                "us_mean": round(c_mean * 1000, 1)}) + "\n")
    except Exception as e:
        print(f"[baseline] torch.compile failed: {e}")

    configs = wmma3_configs()
    print(f"[sweep] {len(configs)} wmma3 configs", flush=True)

    budget_s = args.budget_hours * 3600
    t_start = time.time()
    best_us = eager_us
    best_name = "eager"
    n_correct = 0

    def run_one(BM, BN, BK, WM, WN):
        """Run a single config in this process. Returns result dict or None on crash."""
        name = kernels_wmma3.wmma3_name(BM, BN, BK, WM, WN)
        src = kernels_wmma3.make_kernel_wmma3(BM, BN, BK, WM, WN)
        try:
            res = build_and_test(src, f"torch::Tensor {name}(torch::Tensor A, torch::Tensor B);",
                                 name, A, B, ref, str(build_root / name),
                                 args.trials, rel_tol=args.rel_tol)
        except Exception as e:
            res = {"name": name, "compiled": False, "correct": False,
                   "error": f"crash: {type(e).__name__}: {str(e)[:200]}"}
        return name, res

    for i, (BM, BN, BK, WM, WN) in enumerate(configs):
        if time.time() - t_start > budget_s:
            print(f"[loop] budget exhausted, stopping at config {i}", flush=True)
            break
        name, res = run_one(BM, BN, BK, WM, WN)
        if res.get("us_mean"):
            res["speedup_vs_eager"] = round(eager_us / res["us_mean"], 3)
            if res["us_mean"] < best_us:
                best_us = res["us_mean"]
                best_name = name
                print(f"[BEST] {name}: {res['us_mean']} us "
                      f"({res['speedup_vs_eager']}x vs eager)", flush=True)
        if res["correct"]:
            n_correct += 1
        status = "OK" if res["correct"] else "X"
        print(f"[{i+1:>3}/{len(configs)}] {name:<32} {status} "
              f"us={res.get('us_mean','-'):>7} "
              f"sp={res.get('speedup_vs_eager','-'):>6} "
              f"err={str(res.get('error',''))[:50] or str(res.get('max_diff',''))[:16]}",
              flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"[loop] done: {n_correct}/{len(configs)} correct")
    print(f"[loop] best: {best_name} at {best_us:.1f} us "
          f"({eager_us/best_us:.3f}x vs eager)")


if __name__ == "__main__":
    main()
