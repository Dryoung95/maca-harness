#!/usr/bin/env python3
"""Quick A/B test for the wmma4 fp16-input variant at n=4096."""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kernels_wmma4  # noqa: E402
from loop import build_and_test, time_kernel_cold  # noqa: E402


def main():
    configs = [
        (128, 256, 16, 4, 4),
        (128, 256, 32, 4, 4),
        (128, 256, 32, 4, 2),
        (128, 256, 64, 8, 2),
        (256, 128, 32, 4, 4),
        (256, 256, 32, 8, 4),
        (128, 256, 16, 4, 2),
        (256, 128, 64, 4, 4),
    ]
    out_path = HERE / "wmma4-results.jsonl"
    build_root = HERE / "wmma4build"
    build_root.mkdir(exist_ok=True)

    torch.cuda.set_device(0)
    n = 4096
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B
    Ah, Bh = A.half(), B.half()

    e_mean, e_min, _ = time_kernel_cold(lambda a, b: a @ b, [A, B], num_trials=30)
    eager_us = e_mean * 1000
    print(f"[baseline] eager fp32: {eager_us:.1f} us ({e_min*1000:.1f} min)")
    with open(out_path, "a") as f:
        f.write(json.dumps({"name": "eager_matmul", "correct": True,
                            "us_mean": round(eager_us, 1),
                            "us_min": round(e_min * 1000, 1)}) + "\n")

    for (BM, BN, BK, WM, WN) in configs:
        name = kernels_wmma4.wmma4_name(BM, BN, BK, WM, WN)
        src = kernels_wmma4.make_kernel_wmma4(BM, BN, BK, WM, WN)
        decl = f"torch::Tensor {name}(torch::Tensor A, torch::Tensor B);"
        try:
            res = build_and_test(src, decl, name, Ah, Bh, ref,
                                 str(build_root / name), 30, rel_tol=5e-3)
        except Exception as e:
            res = {"name": name, "compiled": False, "correct": False,
                   "error": f"crash: {type(e).__name__}: {str(e)[:200]}"}
        if res.get("us_mean"):
            res["speedup_vs_eager"] = round(eager_us / res["us_mean"], 3)
        status = "OK" if res.get("correct") else "FAIL"
        print(f"[wmma4] {name:<32} {status} us={res.get('us_mean','-'):>7} "
              f"sp={res.get('speedup_vs_eager','-'):>6} "
              f"err={str(res.get('error',''))[:60] or str(res.get('max_diff',''))[:20]}",
              flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
