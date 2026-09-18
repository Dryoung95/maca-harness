#!/usr/bin/env python3
"""Independent re-verification of the wmma3/wmma6/wmma7 lineages.

The 258-config wmma3 sweep used check_correct(rel_tol=5e-3) against a fp32
reference. Memory notes that store_matrix_sync is a no-op on this toolchain,
yet the sweep logged 276/258 as correct. Re-measure the actual error here to
find out which claim holds, and whether the 'correct' flag is trustworthy.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kernels_wmma3
import kernels_wmma6
import kernels_wmma7
from loop import build_and_test, time_kernel_cold


def describe(out, ref, name):
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    scale = ref.abs().max().item()
    # elementwise relative where reference is non-trivial
    nz = ref.abs() > 0.01 * scale
    rel = (diff[nz] / ref[nz].abs()).max().item() if nz.any() else float("nan")
    print(f"  {name:<34} max_abs={max_abs:10.4f}  scale={scale:8.2f}  "
          f"max_abs/scale={max_abs/scale:.3e}  elem_rel={rel:.3e}")
    return max_abs / scale


def main():
    torch.cuda.set_device(0)
    n = 4096
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B
    Ah, Bh = A.half(), B.half()
    ref_h = (Ah.float() @ Bh.float())
    print(f"setup done, n={n}")

    cases = [
        ("wmma3_128_256_16_4_4", kernels_wmma3, (128, 256, 16, 4, 4), A, B, ref),
        ("wmma6_64_256_32_2_4",  kernels_wmma6, (64, 256, 32, 2, 4), Ah, Bh, ref_h),
        ("wmma7_256_256_16_4_4", kernels_wmma7, (256, 256, 16, 4, 4), Ah, Bh, ref_h),
    ]

    for label, mod, cfg, X, Y, r in cases:
        BM, BN, BK, WM, WN = cfg
        name = f"mm_{label}"
        src = mod.make_kernel_wmma3(*cfg) if label.startswith("wmma3") else \
              (mod.make_kernel_wmma6(*cfg) if label.startswith("wmma6")
               else mod.make_kernel_wmma7(*cfg))
        decl = f"torch::Tensor {name}(torch::Tensor A, torch::Tensor B);"
        print(f"\n=== {name} ===")
        try:
            res = build_and_test(src, decl, name, X, Y, r,
                                 str(HERE / "rebuild" / name), 5, rel_tol=1e9)
        except Exception as e:
            print(f"  BUILD/CRASH: {type(e).__name__}: {str(e)[:200]}")
            continue
        if not res.get("compiled"):
            print(f"  compile failed: {str(res.get('error'))[:200]}")
            continue
        out = None
        try:
            # re-run to inspect the numeric output directly
            from torch.utils.cpp_extension import load_inline
            built = load_inline(name=name, cpp_sources=[decl], cuda_sources=[src],
                                functions=[name],
                                build_directory=str(HERE / "rebuild" / name),
                                verbose=False)
            out = getattr(built, name)(X, Y)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  reload failed: {type(e).__name__}: {str(e)[:150]}")
        if out is not None:
            describe(out, r, name + " vs-ref")
            if res.get("us_mean"):
                print(f"  timing (5 trials): {res['us_mean']} us")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
