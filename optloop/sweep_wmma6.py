#!/usr/bin/env python3
"""Sweep wmma6 tile configs at n=4096, looking for anything that beats eager."""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kernels_wmma6  # noqa: E402
from loop import build_and_test, time_kernel_cold  # noqa: E402


def valid_configs():
    """Yield (BM,BN,BK,WM,WN) with sane thread counts and shared limits."""
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            for BK in (16, 32):
                smem = (BM * (BK + 8) + BK * (BN + 8)) * 2
                if smem > 65536:
                    continue
                for WM in (1, 2, 4):
                    for WN in (1, 2, 4):
                        if BM % (16 * WM) or BN % (16 * WN):
                            continue
                        nwarps = (BM // (16 * WM)) * (BN // (16 * WN))
                        nthreads = 64 * nwarps
                        if nthreads < 64 or nthreads > 1024:
                            continue
                        yield (BM, BN, BK, WM, WN)


def main():
    out_path = HERE / "wmma6-results.jsonl"
    build_root = HERE / "wmma6build"
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
    print(f"[baseline] eager fp32: {eager_us:.1f} us ({e_min*1000:.1f} min)", flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps({"name": "eager_matmul", "correct": True,
                            "us_mean": round(eager_us, 1),
                            "us_min": round(e_min * 1000, 1)}) + "\n")

    configs = list(valid_configs())
    print(f"[sweep] {len(configs)} configs", flush=True)
    best = (eager_us, "eager")

    for i, (BM, BN, BK, WM, WN) in enumerate(configs):
        name = kernels_wmma6.wmma6_name(BM, BN, BK, WM, WN)
        src = kernels_wmma6.make_kernel_wmma6(BM, BN, BK, WM, WN)
        try:
            res = build_and_test(src, f"torch::Tensor {name}(torch::Tensor A, torch::Tensor B);",
                                 name, Ah, Bh, ref, str(build_root / name),
                                 30, rel_tol=5e-3)
        except Exception as e:
            res = {"name": name, "compiled": False, "correct": False,
                   "error": f"crash: {type(e).__name__}: {str(e)[:150]}"}
        if res.get("us_mean"):
            res["speedup_vs_eager"] = round(eager_us / res["us_mean"], 3)
            if res["us_mean"] < best[0]:
                best = (res["us_mean"], name)
                print(f"[BEST] {name}: {res['us_mean']} us "
                      f"({res['speedup_vs_eager']}x)", flush=True)
        status = "OK" if res.get("correct") else "X"
        print(f"[{i+1:>3}/{len(configs)}] {name:<32} {status} "
              f"us={res.get('us_mean','-'):>7} "
              f"sp={res.get('speedup_vs_eager','-'):>6} "
              f"err={str(res.get('error',''))[:40]}", flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        torch.cuda.empty_cache()

    print(f"[sweep] best: {best[1]} at {best[0]:.1f} us "
          f"({eager_us/best[0]:.3f}x vs eager)")


if __name__ == "__main__":
    main()
