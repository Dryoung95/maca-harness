#!/usr/bin/env python3
"""Probe where the wmma matmul's time actually goes.

A/B (ab-inputs-results) and bf16 sweep both show every wmma variant at
0.16-0.68x of eager, independent of input dtype. So input conversion is not
the binding constraint. The remaining suspects:

  1. global->shared tile load (scalar shared writes, known MACA int4-store bug)
  2. the wmma compute itself (16x16x16 mma throughput on C500)
  3. fp32 output store (one scalar global store per element)

Each probe below isolates one stage by removing it and re-timing. If removing
a stage barely changes the time, that stage is not the bottleneck.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loop import build_and_test, time_kernel_cold  # noqa: E402

BM, BN, BK, WM, WN = 128, 256, 16, 4, 4
NTHREADS = 512


def base_src():
    """fp16 inputs, library store, library load — the V2 variant (0.387x)."""
    return """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

constexpr int BM = 128, BN = 256, BK = 16, WM = 4, WN = 4;
constexpr int NWARPS_M = 2, NWARPS_N = 4, NWARPS = 8, NTHREADS = 512;

__global__ void mm_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                          float* __restrict__ C, int n) {
    __shared__ __half sA[BM][BK + 8];
    __shared__ __half sB[BK][BN + 8];
    const int tid = threadIdx.x;
    const int warpId = tid >> 6;
    const int warpM = warpId / NWARPS_N;
    const int warpN = warpId % NWARPS_N;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM][WN];
    #pragma unroll
    for (int i = 0; i < WM; i++)
        #pragma unroll
        for (int j = 0; j < WN; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int numKTiles = (n + BK - 1) / BK;
    for (int kt = 0; kt < numKTiles; kt++) {
        #pragma unroll
        for (int rep = 0; rep < 1; rep++) {
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BK, c = idx % BK;
            int gr = rowBase + r, gc = kt * BK + c;
            if (r < BM && idx < BM * BK) {
                const __half* ptr = &A[(size_t)gr * n + gc];
                __half h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = __float2half(0.0f);
                if (!(gr < n && gc + 1 < n)) h1 = __float2half(0.0f);
                if (!(gr < n && gc + 2 < n)) h2 = __float2half(0.0f);
                if (!(gr < n && gc + 3 < n)) h3 = __float2half(0.0f);
                sA[r][c + 0] = h0; sA[r][c + 1] = h1;
                sA[r][c + 2] = h2; sA[r][c + 3] = h3;
            }
        }
        #pragma unroll
        for (int rep = 0; rep < 2; rep++) {
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BN, c = idx % BN;
            int gr = kt * BK + r, gc = colBase + c;
            if (r < BK && idx < BK * BN) {
                const __half* ptr = &B[gr * n + gc];
                __half h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = __float2half(0.0f);
                if (!(gr < n && gc + 1 < n)) h1 = __float2half(0.0f);
                if (!(gr < n && gc + 2 < n)) h2 = __float2half(0.0f);
                if (!(gr < n && gc + 3 < n)) h3 = __float2half(0.0f);
                sB[r][c + 0] = h0; sB[r][c + 1] = h1;
                sB[r][c + 2] = h2; sB[r][c + 3] = h3;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[WM];
            #pragma unroll
            for (int i = 0; i < WM; i++)
                wmma::load_matrix_sync(a[i],
                    &sA[warpM * (16 * WM) + i * 16][kk], BK + 8);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[WN];
            #pragma unroll
            for (int j = 0; j < WN; j++)
                wmma::load_matrix_sync(b[j],
                    &sB[kk][warpN * (16 * WN) + j * 16], BN + 8);
            #pragma unroll
            for (int i = 0; i < WM; i++)
                #pragma unroll
                for (int j = 0; j < WN; j++)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < WM; i++) {
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {
            int gc = colBase + warpN * (16 * WN) + j * 16;
            if (gr + 15 < n && gc + 15 < n)
                wmma::store_matrix_sync(&C[(size_t)gr * n + gc], acc[i][j],
                                        (unsigned)n, wmma::mem_row_major);
        }
    }
}

torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options().dtype(at::kFloat));
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), n);
    return C;
}
"""


def probe_no_compute():
    """Same memory traffic (load tiles + store output), zero mma work.

    Result vs full kernel = cost of the compute stage alone.
    """
    s = base_src().replace(
        "wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);",
        ";  // compute removed")
    return s


def probe_no_load():
    """mma on shared tiles that are never filled (all zeros)."""
    s = base_src()
    # drop both load loops by neutralizing the guards
    s = s.replace("if (r < BM && idx < BM * BK) {", "if (false) {")
    s = s.replace("if (r < BK && idx < BK * BN) {", "if (false) {")
    return s


def probe_no_store():
    """compute + load, output never written."""
    s = base_src().replace(
        "wmma::store_matrix_sync(&C[(size_t)gr * n + gc], acc[i][j],",
        "; if (false) wmma::store_matrix_sync(&C[(size_t)gr * n + gc], acc[i][j],")
    return s


def main():
    torch.cuda.set_device(0)
    n = 4096
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B
    Ah, Bh = A.half(), B.half()
    ref_h = Ah.float() @ Bh.float()

    e, _, _ = time_kernel_cold(lambda a, b: a @ b, [Ah, Bh], num_trials=25)
    eager_h = e * 1000
    print(f"[baseline] eager fp16: {eager_h:.1f} us", flush=True)

    probes = [
        ("full_kernel", base_src(), ref_h),
        ("no_compute", probe_no_compute(), ref_h),
        ("no_load", probe_no_load(), None),
        ("no_store", probe_no_store(), None),
    ]
    out_path = HERE / "probe-bound-results.jsonl"
    for label, src, r in probes:
        name = f"probe_{label}"
        print(f"\n=== {label} ===", flush=True)
        # probes that change results are not checked for correctness
        res = build_and_test(src,
                             "torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B);",
                             name, Ah, Bh, ref_h,
                             str(HERE / "probebuild" / name), 25, rel_tol=1e9)
        res["label"] = label
        res["eager_fp16_us"] = round(eager_h, 1)
        if res.get("us_mean"):
            res["speedup_vs_eager_fp16"] = round(eager_h / res["us_mean"], 3)
            print(f"  -> {res['us_mean']} us  {res['speedup_vs_eager_fp16']}x vs eager fp16",
                  flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
