#!/usr/bin/env python3
"""A/B experiment: is the wmma3 line slow because of fp32 inputs, or because
of the library wmma load/store primitives?

Two confounded claims need separating:
  (a) wmma3 uses fp32 inputs + per-element fp32->fp16 conversion in the K loop
      (profile REPORT.md calls this the root cause of the 0.682x result);
  (b) memory notes say load_matrix_sync / store_matrix_sync are broken on
      MACA 3.3.0, forcing the hand-written load/store of wmma6/wmma7.
Claim (b) is contradicted by verify_all.py: wmma3 reaches end-to-end
max_abs/scale = 2.8e-05 through the library store. That is fp16 rounding, not
a no-op store.

Builds one tile config (BM=128,BN=256,BK=16,WM=4,WN=4) in four variants and
times each with cuda_event:
  V1 fp32 inputs, library store   (wmma3 as swept)
  V2 fp16 inputs, library store   (isolates claim (a))
  V3 fp16 inputs, hand store      (isolates claim (b))
  V4 fp32 inputs, hand store      (both changes)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loop import build_and_test, time_kernel_cold  # noqa: E402

BM, BN, BK, WM, WN = 128, 256, 16, 4, 4
NWARP_M = BM // (16 * WM)
NWARP_N = BN // (16 * WN)
NWARPS = NWARP_M * NWARP_N
NTHREADS = 64 * NWARPS
A_LOADS = (BM * BK + 4 * NTHREADS - 1) // (4 * NTHREADS)
B_LOADS = (BK * BN + 4 * NTHREADS - 1) // (4 * NTHREADS)

HEAD = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 16;
constexpr int WM = 4;
constexpr int WN = 4;
constexpr int NWARPS_M = 2;
constexpr int NWARPS_N = 4;
constexpr int NWARPS = 8;
constexpr int NTHREADS = 512;

__device__ inline void store_acc_16x16(
        float* C, unsigned ldm,
        const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& f) {
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        C[(row + i) * ldm + col] = f.x[i];
}
"""

LOAD_A_FP32 = """
        #pragma unroll
        for (int rep = 0; rep < 1; rep++) {
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BK;
            int c = idx % BK;
            int gr = rowBase + r;
            int gc = kt * BK + c;
            if (r < BM && idx < BM * BK) {
                int4 v = *reinterpret_cast<const int4*>(&A[(size_t)gr * n + gc]);
                float f0 = __int_as_float(v.x), f1 = __int_as_float(v.y);
                float f2 = __int_as_float(v.z), f3 = __int_as_float(v.w);
                if (!(gr < n && gc + 0 < n)) f0 = 0.0f;
                if (!(gr < n && gc + 1 < n)) f1 = 0.0f;
                if (!(gr < n && gc + 2 < n)) f2 = 0.0f;
                if (!(gr < n && gc + 3 < n)) f3 = 0.0f;
                sA[r][c + 0] = __float2half(f0);
                sA[r][c + 1] = __float2half(f1);
                sA[r][c + 2] = __float2half(f2);
                sA[r][c + 3] = __float2half(f3);
            }
        }
"""

LOAD_B_FP32 = """
        #pragma unroll
        for (int rep = 0; rep < 2; rep++) {
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BN;
            int c = idx % BN;
            int gr = kt * BK + r;
            int gc = colBase + c;
            if (r < BK && idx < BK * BN) {
                int4 v = *reinterpret_cast<const int4*>(&B[gr * n + gc]);
                float f0 = __int_as_float(v.x), f1 = __int_as_float(v.y);
                float f2 = __int_as_float(v.z), f3 = __int_as_float(v.w);
                if (!(gr < n && gc + 0 < n)) f0 = 0.0f;
                if (!(gr < n && gc + 1 < n)) f1 = 0.0f;
                if (!(gr < n && gc + 2 < n)) f2 = 0.0f;
                if (!(gr < n && gc + 3 < n)) f3 = 0.0f;
                sB[r][c + 0] = __float2half(f0);
                sB[r][c + 1] = __float2half(f1);
                sB[r][c + 2] = __float2half(f2);
                sB[r][c + 3] = __float2half(f3);
            }
        }
"""

LOAD_A_FP16 = """
        #pragma unroll
        for (int rep = 0; rep < 1; rep++) {
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BK;
            int c = idx % BK;
            int gr = rowBase + r;
            int gc = kt * BK + c;
            if (r < BM && idx < BM * BK) {
                const __half* ptr = &A[(size_t)gr * n + gc];
                __half h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = __float2half(0.0f);
                if (!(gr < n && gc + 1 < n)) h1 = __float2half(0.0f);
                if (!(gr < n && gc + 2 < n)) h2 = __float2half(0.0f);
                if (!(gr < n && gc + 3 < n)) h3 = __float2half(0.0f);
                sA[r][c + 0] = h0;
                sA[r][c + 1] = h1;
                sA[r][c + 2] = h2;
                sA[r][c + 3] = h3;
            }
        }
"""

LOAD_B_FP16 = """
        #pragma unroll
        for (int rep = 0; rep < 2; rep++) {
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BN;
            int c = idx % BN;
            int gr = kt * BK + r;
            int gc = colBase + c;
            if (r < BK && idx < BK * BN) {
                const __half* ptr = &B[gr * n + gc];
                __half h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = __float2half(0.0f);
                if (!(gr < n && gc + 1 < n)) h1 = __float2half(0.0f);
                if (!(gr < n && gc + 2 < n)) h2 = __float2half(0.0f);
                if (!(gr < n && gc + 3 < n)) h3 = __float2half(0.0f);
                sB[r][c + 0] = h0;
                sB[r][c + 1] = h1;
                sB[r][c + 2] = h2;
                sB[r][c + 3] = h3;
            }
        }
"""

COMPUTE = """
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
"""

STORE_LIB = """
    #pragma unroll
    for (int i = 0; i < WM; i++) {
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {
            int gc = colBase + warpN * (16 * WN) + j * 16;
            if (gr + 15 < n && gc + 15 < n) {
                wmma::store_matrix_sync(&C[(size_t)gr * n + gc], acc[i][j],
                                        (unsigned)n, wmma::mem_row_major);
            } else {
                float tmp[16];
                wmma::store_matrix_sync(tmp, acc[i][j], 16, wmma::mem_row_major);
                for (int rr = 0; rr < 16 && gr + rr < n; rr++)
                    for (int cc = 0; cc < 16 && gc + cc < n; cc++)
                        C[(size_t)(gr + rr) * n + gc + cc] = tmp[rr * 16 + cc];
            }
        }
    }
"""

STORE_HAND = """
    #pragma unroll
    for (int i = 0; i < WM; i++) {
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {
            int gc = colBase + warpN * (16 * WN) + j * 16;
            if (gr + 15 < n && gc + 15 < n) {
                store_acc_16x16(&C[(size_t)gr * n + gc], (unsigned)n, acc[i][j]);
            } else {
                __shared__ float stmp[16 * 16];
                if (warpId < NWARPS) {
                    unsigned row = (warpId >> 4) << 2;
                    unsigned col = warpId & 0xf;
                    #pragma unroll
                    for (int t = 0; t < 4; t++)
                        stmp[(row + t) * 16 + col] = acc[i][j].x[t];
                }
                __syncthreads();
                for (int rr = 0; rr < 16 && gr + rr < n; rr++)
                    for (int cc = 0; cc < 16 && gc + cc < n; cc++)
                        C[(size_t)(gr + rr) * n + gc + cc] = stmp[rr * 16 + cc];
            }
        }
    }
"""


def kernel_source(use_fp16_inputs, use_hand_store):
    if use_fp16_inputs:
        load_a, load_b = LOAD_A_FP16, LOAD_B_FP16
        ptr = "const __half* __restrict__"
        a_arg = "reinterpret_cast<const __half*>(A.data_ptr<at::Half>())"
        b_arg = "reinterpret_cast<const __half*>(B.data_ptr<at::Half>())"
        checks = ("    TORCH_CHECK(A.scalar_type() == at::kHalf, \"A must be fp16\");\n"
                  "    TORCH_CHECK(B.scalar_type() == at::kHalf, \"B must be fp16\");\n")
    else:
        load_a, load_b = LOAD_A_FP32, LOAD_B_FP32
        ptr = "const float* __restrict__"
        a_arg = "A.data_ptr<float>()"
        b_arg = "B.data_ptr<float>()"
        checks = ""
    store = STORE_HAND if use_hand_store else STORE_LIB
    src = HEAD + """
__global__ void mm_kernel(""" + ptr + """ A, """ + ptr + """ B, float* __restrict__ C, int n) {
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
""" + load_a + load_b + COMPUTE + store + """
}

torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B) {
""" + checks + """
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options().dtype(at::kFloat));
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_kernel<<<grid, block>>>(A_PTR, B_PTR, C.data_ptr<float>(), n);
    return C;
}
"""
    return src.replace("A_PTR", a_arg).replace("B_PTR", b_arg)


def main():
    torch.cuda.set_device(0)
    n = 4096
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B
    Ah, Bh = A.half(), B.half()

    e_mean, _, _ = time_kernel_cold(lambda a, b: a @ b, [A, B], num_trials=20)
    eager_us = e_mean * 1000
    print(f"[baseline] eager fp32 matmul: {eager_us:.1f} us", flush=True)

    variants = [
        ("V1_fp32_libstore",   False, False),
        ("V2_fp16_libstore",   True,  False),
        ("V3_fp16_handstore",  True,  True),
        ("V4_fp32_handstore",  False, True),
    ]

    out_path = HERE / "ab-inputs-results.jsonl"
    for label, use_fp16, use_hand in variants:
        name = f"ab_{label}"
        X, Y = (Ah, Bh) if use_fp16 else (A, B)
        r = (Ah.float() @ Bh.float()) if use_fp16 else ref
        src = kernel_source(use_fp16, use_hand)
        print(f"\n=== {label} ===", flush=True)
        try:
            res = build_and_test(src,
                                 "torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B);",
                                 name, X, Y, r, str(HERE / "abbuild" / name),
                                 25, rel_tol=5e-3)
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {str(e)[:200]}", flush=True)
            res = {"name": name, "error": f"crash {type(e).__name__}"}
        res["label"] = label
        res["eager_us"] = round(eager_us, 1)
        if res.get("us_mean"):
            res["speedup_vs_eager"] = round(eager_us / res["us_mean"], 3)
            print(f"  -> {res['us_mean']} us  {res['speedup_vs_eager']}x vs eager", flush=True)
        if res.get("max_diff") is not None:
            print(f"  -> INCORRECT max_diff={res['max_diff']:.4g}", flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
