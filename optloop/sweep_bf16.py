#!/usr/bin/env python3
"""Sweep wmma matmul with bf16 inputs + fp32 accumulator.

A/B results (ab-inputs-results) showed fp16 *inputs* are SLOWER than fp32
inputs at the same tile (0.387x vs 0.684x), which inverts the profile
REPORT.md hypothesis that the in-loop fp32->fp16 conversion was the root
cause. eager bf16 matmul is the fastest path on C500 (730 us vs fp16 881 vs
fp32 1587), so the tensor-core units clearly run best on bf16.

Both fp16 variants still write fp32 output, which costs one scalar global
store per element. This sweep tests bf16 inputs against both fp32 and bf16
output dtypes to separate input-format effects from output-store cost.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loop import build_and_test, time_kernel_cold  # noqa: E402

BM, BN, BK = 128, 256, 16
NTHREADS = 512
NWARP_M, NWARP_N = 2, 4
NWARPS = 8


def kernel_source(in_dtype, out_dtype):
    """in_dtype: 'bf16' | 'fp16';  out_dtype: 'fp32' | 'bf16' | 'fp16'."""
    if in_dtype == "bf16":
        frag_t = "maca_bfloat16"
        in_c = "__nv_bfloat16"
        in_ptr = "const __nv_bfloat16* __restrict__"
        in_arg = "reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>())"
        in_arg_b = "reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>())"
        zero = "__float2bfloat16(0.0f)"
    else:
        frag_t = "__half"
        in_c = "__half"
        in_ptr = "const __half* __restrict__"
        in_arg = "reinterpret_cast<const __half*>(A.data_ptr<at::Half>())"
        in_arg_b = "reinterpret_cast<const __half*>(B.data_ptr<at::Half>())"
        zero = "__float2half(0.0f)"

    if out_dtype == "fp32":
        out_c = "float"
        out_ptr = "float* __restrict__"
        out_arg = "C.data_ptr<float>()"
        out_empty = "at::kFloat"
        store = """
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
"""
    else:
        out_c = in_c
        out_ptr = in_ptr
        if out_dtype == "bf16":
            out_arg = "reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>())"
            out_empty = "at::kBFloat16"
        else:
            out_arg = "reinterpret_cast<__half*>(C.data_ptr<at::Half>())"
            out_empty = "at::kHalf"
        # fp16/bf16 output: convert each accumulator element then store scalar
        store = """
    #pragma unroll
    for (int i = 0; i < WM; i++) {
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {
            int gc = colBase + warpN * (16 * WN) + j * 16;
            if (gr + 15 < n && gc + 15 < n) {
                __shared__ {OUTT} stmp[16 * 16];
                unsigned row = (__lane_id() >> 4) << 2;
                unsigned col = __lane_id() & 0xf;
                #pragma unroll
                for (int t = 0; t < 4; t++)
                    stmp[(row + t) * 16 + col] = ({OUTT})acc[i][j].x[t];
                __syncthreads();
                #pragma unroll
                for (int rr = 0; rr < 16; rr += 4) {
                    int4 v;
                    v.x = __bfloat16_as_short(stmp[(row/4)*0 + 0*0 + rr*16 + col]) * 0;
                    v.x = 0;
                    // scalar fallback: 4 stores per thread
                }
                for (int rr = 0; rr < 16; rr++)
                    for (int cc = 0; cc < 16; cc++)
                        C[(size_t)(gr + rr) * n + gc + cc] = stmp[rr * 16 + cc];
            }
        }
    }
""".replace("{OUTT}", in_c)
    return f"""
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

__global__ void mm_kernel({in_ptr} A, {in_ptr} B, {out_ptr} C, int n) {{
    __shared__ {frag_t} sA[BM][BK + 8];
    __shared__ {frag_t} sB[BK][BN + 8];

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

    for (int kt = 0; kt < numKTiles; kt++) {{
        #pragma unroll
        for (int rep = 0; rep < 1; rep++) {{
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BK;
            int c = idx % BK;
            int gr = rowBase + r;
            int gc = kt * BK + c;
            if (r < BM && idx < BM * BK) {{
                const {in_c}* ptr = &A[(size_t)gr * n + gc];
                {in_c} h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = {zero};
                if (!(gr < n && gc + 1 < n)) h1 = {zero};
                if (!(gr < n && gc + 2 < n)) h2 = {zero};
                if (!(gr < n && gc + 3 < n)) h3 = {zero};
                sA[r][c + 0] = h0;
                sA[r][c + 1] = h1;
                sA[r][c + 2] = h2;
                sA[r][c + 3] = h3;
            }}
        }}
        #pragma unroll
        for (int rep = 0; rep < 2; rep++) {{
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BN;
            int c = idx % BN;
            int gr = kt * BK + r;
            int gc = colBase + c;
            if (r < BK && idx < BK * BN) {{
                const {in_c}* ptr = &B[gr * n + gc];
                {in_c} h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = {zero};
                if (!(gr < n && gc + 1 < n)) h1 = {zero};
                if (!(gr < n && gc + 2 < n)) h2 = {zero};
                if (!(gr < n && gc + 3 < n)) h3 = {zero};
                sB[r][c + 0] = h0;
                sB[r][c + 1] = h1;
                sB[r][c + 2] = h2;
                sB[r][c + 3] = h3;
            }}
        }}
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {{
            wmma::fragment<wmma::matrix_a, 16, 16, 16, {frag_t}, wmma::row_major> a[WM];
            #pragma unroll
            for (int i = 0; i < WM; i++)
                wmma::load_matrix_sync(a[i],
                    &sA[warpM * (16 * WM) + i * 16][kk], BK + 8);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, {frag_t}, wmma::row_major> b[WN];
            #pragma unroll
            for (int j = 0; j < WN; j++)
                wmma::load_matrix_sync(b[j],
                    &sB[kk][warpN * (16 * WN) + j * 16], BN + 8);
            #pragma unroll
            for (int i = 0; i < WM; i++)
                #pragma unroll
                for (int j = 0; j < WN; j++)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }}
        __syncthreads();
    }}
{store}
}}

torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options().dtype({out_empty}));
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_kernel<<<grid, block>>>({in_arg}, {in_arg_b}, {out_arg}, n);
    return C;
}}
"""


def main():
    torch.cuda.set_device(0)
    n = 4096
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B
    Ab, Bb = A.bfloat16(), B.bfloat16()
    Ah, Bh = A.half(), B.half()

    e, _, _ = time_kernel_cold(lambda a, b: a @ b, [Ab, Bb], num_trials=25)
    eager_bf16 = e * 1000
    print(f"[baseline] eager bf16: {eager_bf16:.1f} us", flush=True)

    cases = [
        ("bf16_in_fp32_out", "bf16", "fp32", Ab, Bb, ref),
        ("fp16_in_fp32_out", "fp16", "fp32", Ah, Bh, ref),
    ]
    out_path = HERE / "bf16-results.jsonl"
    for label, ind, outd, X, Y, r in cases:
        name = f"bf_{label}"
        src = kernel_source(ind, outd)
        print(f"\n=== {label} ===", flush=True)
        try:
            res = build_and_test(
                src, "torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B);",
                name, X, Y, r, str(HERE / "bf16build" / name), 25, rel_tol=5e-3)
        except Exception as ex:
            print(f"  CRASH: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
            res = {"name": name, "error": f"crash {type(ex).__name__}"}
        res["label"] = label
        res["eager_bf16_us"] = round(eager_bf16, 1)
        if res.get("us_mean"):
            res["speedup_vs_eager_bf16"] = round(eager_bf16 / res["us_mean"], 3)
            print(f"  -> {res['us_mean']} us  {res['speedup_vs_eager_bf16']}x vs eager bf16",
                  flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
