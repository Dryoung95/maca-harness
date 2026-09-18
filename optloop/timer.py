#!/usr/bin/env python3
"""
C500 optimization loop for KernelBench L1 P1 (square matmul, N=4096).

Design: an iterative kernel-optimization harness that
  1. compiles a candidate CUDA kernel via load_inline (cu-bridge -> cucc -> mxcc)
  2. verifies correctness against torch.matmul (allclose, fp32 tol 1e-4..1e-3)
  3. times it with cuda_event (cold-cache, warmup, multiple trials)
  4. records (kernel_id, params, correctness, us, speedup_vs_eager, notes)

Long-running: each iteration is one (compile + verify + time) cycle. The loop
runs for a wall-clock budget and writes results incrementally so partial
progress always survives.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1] / "KernelBench"
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------- timing

def time_kernel(fn, args, num_warmup=10, num_trials=50):
    """cuda_event timing in ms -> (mean, min, std)."""
    device = torch.device("cuda:0")
    for _ in range(num_warmup):
        fn(*args)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize(device)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize(device)
        times.append(s.elapsed_time(e))
    t = torch.tensor(times)
    return float(t.mean().item()), float(t.min().item()), float(t.std().item())


def l2_trash(device):
    dummy = torch.empty(2 * 1024 * 1024, dtype=torch.int8, device=device)
    dummy.fill_(42)
    del dummy


def time_kernel_cold(fn, args, num_warmup=10, num_trials=30):
    """Timing with L2 flush before each trial (KernelBench-style cold cache)."""
    device = torch.device("cuda:0")
    for _ in range(num_warmup):
        fn(*args)
        torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize(device)
        l2_trash(device)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize(device)
        times.append(s.elapsed_time(e))
    t = torch.tensor(times)
    return float(t.mean().item()), float(t.min().item()), float(t.std().item())


# ---------------------------------------------------------------- kernels

# Candidate 0: the upstream few-shot tiled kernel (16x16 tiles), baseline.
KERNEL_TILED16 = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void mm_tiled_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = (row < n && t * TILE + threadIdx.x < n) ? A[row * n + t * TILE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (t * TILE + threadIdx.y < n && col < n) ? B[(t * TILE + threadIdx.y) * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = acc;
}

torch::Tensor mm_tiled(torch::Tensor A, torch::Tensor B) {
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options());
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    mm_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}
"""

# Candidate generator: parametrized tiled matmul with
#  - tile size T (per-thread output block)
#  - threads per block arranged as (T, BM) work-per-thread along N
#  - vectorized float4 loads
#  - K-dimension unrolling
def make_tiled_kernel(T: int, BM: int, BN: int, UNROLL: int, VEC: int):
    """
    Tiled kernel: each block computes a BM x BN output tile.
    Threads: (BM/T) x (BN/T) each computing a T x T micro-tile? Simplified:
    We use a straightforward parametrization:
      block = (BN/T, BM/T) threads? keep it simple: thread block (BM, BN/T)?
    Simplest robust scheme: BM x BN tile, threads = (BM, BN/T) each doing T outputs.
    """
    threads_x = BN // T
    threads_y = BM
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int T = {T};
constexpr int UNROLL = {UNROLL};
constexpr int VEC = {VEC};

__global__ void mm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {{
    __shared__ float sA[BM][BN + 8];   // padding to reduce bank conflicts (warp 64)
    __shared__ float sB[BM][BN + 8];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx * T;
    float acc[T];
    #pragma unroll
    for (int i = 0; i < T; i++) acc[i] = 0.0f;

    for (int kt = 0; kt < (n + BN - 1) / BN; kt++) {{
        // load BM x BN tile of A and B into shared
        for (int i = 0; i < BM; i += 1) {{
            int r = blockIdx.y * BM + i;
            int k = kt * BN + tx;
            sA[i][tx] = (r < n && k < n) ? A[r * n + k] : 0.0f;
        }}
        for (int j = 0; j < BN; j += 1) {{
            int k = kt * BN + ty;
            int c = blockIdx.x * BN + j;
            sB[ty][j] = (k < n && c < n) ? B[k * n + c] : 0.0f;
        }}
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BN; i += UNROLL) {{
            #pragma unroll
            for (int tt = 0; tt < T; tt++) {{
                #pragma unroll
                for (int u = 0; u < UNROLL; u++) {{
                    if (i + u < BN) acc[tt] += sA[ty][i + u] * sB[i + u][tx * T + tt];
                }}
            }}
        }}
        __syncthreads();
    }}
    #pragma unroll
    for (int tt = 0; tt < T; tt++) {{
        int c = col + tt;
        if (row < n && c < n) C[row * n + c] = acc[tt];
    }}
}}

torch::Tensor mm_{BM}_{BN}_{T}_{UNROLL}_{VEC}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(BN / T, BM);
    mm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}}
"""


# Simplest robust high-performance design for C500 (warp 64, 104 SM):
# 1. Each thread computes multiple outputs (T per thread) -> reduces launch overhead
# 2. Shared memory tiles with +8 column padding (bank conflicts on warp64)
# 3. K-loop unrolling
def make_kernel_v2(BM: int, BN: int, T: int, UNROLL: int):
    """v2: thread block (BM, BN/T); each thread handles T consecutive N-elements."""
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int T = {T};
constexpr int UNROLL = {UNROLL};

__global__ void mm_v2_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {{
    // shared tiles: A tile is BM x BN (transposed loads not needed), B tile is BM x BN
    __shared__ float sA[BM][BN];
    __shared__ float sB[BN][BN];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * BM + ty;
    const int colBase = blockIdx.x * BN + tx * T;

    float acc[T];
    #pragma unroll
    for (int i = 0; i < T; i++) acc[i] = 0.0f;

    const int numKTiles = (n + BN - 1) / BN;
    for (int kt = 0; kt < numKTiles; kt++) {{
        // cooperative load of A tile: BM rows, BN cols -> each thread loads BN/T? too few threads
        // threads = (BM, BN/T). Each thread loads one A element per inner step.
        // A tile: BM x BN: thread (ty,tx) loads A[row, kt*BN + tx + (BN/T)*?] -> need BN/(BN/T) = T loads
        #pragma unroll
        for (int rep = 0; rep < T; rep++) {{
            int kcol = kt * BN + tx + rep * (BN / T);
            if (kcol < n) sA[ty][tx + rep * (BN / T)] = A[row * n + kcol];
            else sA[ty][tx + rep * (BN / T)] = 0.0f;
        }}
        // B tile: BN x BN: thread (ty,tx) loads T elements down column (tx*T + rep)
        #pragma unroll
        for (int rep = 0; rep < T; rep++) {{
            int br = kt * BN + ty + rep * (BM / 1);
            // distribute BN rows across BN/T threads * T
        }}
        // simpler: each thread loads one B element per K-step via grid stride
        for (int rr = 0; rr < BN; rr += (BN / T)) {{
            int br = kt * BN + rr / (BN / T) * 1 + ty;  // placeholder
        }}
        __syncthreads();
        // compute
        #pragma unroll
        for (int i = 0; i < BN; i++) {{
            float a = sA[ty][i];
            #pragma unroll
            for (int tt = 0; tt < T; tt++) {{
                acc[tt] += a * sB[i][tx * T + tt];
            }}
        }}
        __syncthreads();
    }}
    #pragma unroll
    for (int tt = 0; tt < T; tt++) {{
        int c = colBase + tt;
        if (row < n && c < n) C[row * n + c] = acc[tt];
    }}
}}

torch::Tensor mm_v2_{BM}_{BN}_{T}_{UNROLL}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(BN / T, BM);
    mm_v2_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}}
"""


# v3: clean and correct — the reference design we will actually sweep.
# Block computes BM x BN output tile. Threads = (BM, BN/T), each handles T outputs.
# A tile loaded cooperatively: each thread loads T elements along K (stride BN/T... no).
# Correct cooperative load with threads (BM, BN/T):
#   A tile has BM*BN elements; threads = BM*(BN/T); each thread loads T elements.
def make_kernel_v3(BM: int, BN: int, T: int):
    """v3: correct cooperative tiling. threads=(BM, BN/T)."""
    nthreads = BM * (BN // T)
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int T = {T};

__global__ void mm_v3_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {{
    __shared__ float sA[BM][BN + 8];
    __shared__ float sB[BN][BN + 8];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * BM + ty;
    const int colBase = blockIdx.x * BN + tx * T;

    float acc[T];
    #pragma unroll
    for (int i = 0; i < T; i++) acc[i] = 0.0f;

    const int numKTiles = (n + BN - 1) / BN;
    for (int kt = 0; kt < numKTiles; kt++) {{
        // A tile (BM x BN): thread (ty,tx) loads elements [ty][tx + rep*(BN/T)] for rep in 0..T-1
        #pragma unroll
        for (int rep = 0; rep < T; rep++) {{
            int kk = kt * BN + tx + rep * (BN / T);
            float v = 0.0f;
            if (row < n && kk < n) v = A[row * n + kk];
            sA[ty][tx + rep * (BN / T)] = v;
        }}
        // B tile (BN x BN): thread (ty,tx) loads elements [ty + rep*(BN/T)][tx*T .. +T-1]
        #pragma unroll
        for (int rep = 0; rep < T; rep++) {{
            int br = kt * BN + ty + rep * (BN / T);
            #pragma unroll
            for (int tt = 0; tt < T; tt++) {{
                int bc = blockIdx.x * BN + tx * T + tt;
                float v = 0.0f;
                if (br < n && bc < n) v = B[br * n + bc];
                sB[ty + rep * (BN / T)][tx * T + tt] = v;
            }}
        }}
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BN; i++) {{
            float a = sA[ty][i];
            #pragma unroll
            for (int tt = 0; tt < T; tt++) {{
                acc[tt] += a * sB[i][tx * T + tt];
            }}
        }}
        __syncthreads();
    }}
    #pragma unroll
    for (int tt = 0; tt < T; tt++) {{
        int c = colBase + tt;
        if (row < n && c < n) C[row * n + c] = acc[tt];
    }}
}}

torch::Tensor mm_v3_{BM}_{BN}_{T}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(BN / T, BM);
    mm_v3_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}}
"""

CPP = "torch::Tensor mm_v3_{bm}_{bn}_{t}(torch::Tensor A, torch::Tensor B);"


def build_and_test(kernel_src, cpp_src, name, A, B, ref, build_dir, verbose=False):
    """Compile a candidate, verify correctness, time it. Returns result dict."""
    from torch.utils.cpp_extension import load_inline
    os.makedirs(build_dir, exist_ok=True)
    t0 = time.time()
    result = {"name": name, "compiled": False, "correct": False, "us": None, "speedup": None}
    try:
        mod = load_inline(
            name=name,
            cpp_sources=[cpp_src],
            cuda_sources=[kernel_src],
            functions=[name],
            build_directory=build_dir,
            verbose=verbose,
        )
        result["compiled"] = True
        result["compile_seconds"] = round(time.time() - t0, 1)

        fn = getattr(mod, name)
        out = fn(A, B)
        torch.cuda.synchronize()
        ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        result["correct"] = bool(ok)
        if not ok:
            result["max_diff"] = float((out - ref).abs().max().item())
        else:
            mean_ms, min_ms, std_ms = time_kernel_cold(fn, [A, B])
            result["us_mean"] = round(mean_ms * 1000, 1)
            result["us_min"] = round(min_ms * 1000, 1)
            result["us_std"] = round(std_ms * 1000, 1)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/cuda-harness-migration/optloop/results.jsonl")
    ap.add_argument("--build-root", default="/data/cuda-harness-migration/optloop/build")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=4096)
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

    def eager(a, b):
        return a @ b

    e_mean, e_min, e_std = time_kernel_cold(eager, [A, B], num_trials=args.trials)
    eager_us = e_mean * 1000
    print(f"[baseline] eager torch.matmul: {eager_us:.1f} us (min {e_min*1000:.1f}, std {e_std*1000:.1f})")
    base = {"name": "eager_matmul", "compiled": True, "correct": True,
            "us_mean": round(eager_us, 1), "us_min": round(e_min * 1000, 1),
            "us_std": round(e_std * 1000, 1)}
    with open(out_path, "a") as f:
        f.write(json.dumps(base) + "\n")

    # torch.compile reference
    try:
        cmodel = torch.compile(eager)
        for _ in range(5):
            cmodel(A, B)
        c_mean, c_min, c_std = time_kernel_cold(cmodel, [A, B], num_trials=args.trials)
        print(f"[baseline] torch.compile:    {c_mean*1000:.1f} us")
        rec = {"name": "torch_compile", "compiled": True, "correct": True,
               "us_mean": round(c_mean * 1000, 1), "us_min": round(c_min * 1000, 1),
               "us_std": round(c_std * 1000, 1)}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print(f"[baseline] torch.compile failed: {e}")

    # upstream tiled-16 baseline
    res = build_and_test(
        KERNEL_TILED16, "torch::Tensor mm_tiled(torch::Tensor A, torch::Tensor B);",
        "mm_tiled", A, B, ref, str(build_root / "tiled16"))
    res["speedup_vs_eager"] = round(eager_us / res["us_mean"], 3) if res.get("us_mean") else None
    print(f"[cand] {res['name']}: compiled={res['compiled']} correct={res['correct']} "
          f"us={res.get('us_mean')} speedup={res.get('speedup_vs_eager')}")
    with open(out_path, "a") as f:
        f.write(json.dumps(res) + "\n")

    # v3 parameter sweep
    configs = []
    for BM in (32, 64, 128):
        for BN in (32, 64, 128):
            for T in (1, 2, 4, 8):
                nthreads = BM * (BN // T)
                if nthreads > 2048 or nthreads < 64:
                    continue
                smem = (BM * (BN + 8) + BN * (BN + 8)) * 4
                if smem > 65536:
                    continue
                configs.append((BM, BN, T))
    print(f"[sweep] {len(configs)} v3 configs")

    budget_s = args.budget_hours * 3600
    t_start = time.time()
    for i, (BM, BN, T) in enumerate(configs):
        if time.time() - t_start > budget_s:
            print(f"[sweep] budget reached, stopping")
            break
        name = f"mm_v3_{BM}_{BN}_{T}"
        src = make_kernel_v3(BM, BN, T)
        cpp = CPP.format(bm=BM, bn=BN, t=T)
        res = build_and_test(src, cpp, name, A, B, ref, str(build_root / name))
        if res.get("us_mean"):
            res["speedup_vs_eager"] = round(eager_us / res["us_mean"], 3)
        print(f"[cand {i+1}/{len(configs)}] {name}: comp={res['compiled']} corr={res['correct']} "
              f"us={res.get('us_mean')} speedup={res.get('speedup_vs_eager')} err={res.get('error','')[:80]}")
        with open(out_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        torch.cuda.empty_cache()

    print("[done] sweep complete")


if __name__ == "__main__":
    main()
