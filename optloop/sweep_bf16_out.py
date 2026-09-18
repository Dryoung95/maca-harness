#!/usr/bin/env python3
"""bf16 wmma sweep with bf16 output.

bf16 is the fastest eager path on C500 (812-840us vs fp16 889, fp32-TF32
1592). Earlier sweeps only tested fp32 output, where the scalar fp32 store
dominates. This sweeps bf16 in / bf16 out over tile shapes.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loop import build_and_test, time_kernel_cold  # noqa: E402


def make_kernel(BM, BN, BK, WM, WN):
    nwarps_m = BM // (16 * WM)
    nwarps_n = BN // (16 * WN)
    nwarps = nwarps_m * nwarps_n
    nthreads = 64 * nwarps
    a_loads = (BM * BK + 4 * nthreads - 1) // (4 * nthreads)
    b_loads = (BK * BN + 4 * nthreads - 1) // (4 * nthreads)
    parts = []
    parts.append("""#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
""")
    parts.append("constexpr int BM = %d, BN = %d, BK = %d, WM = %d, WN = %d;"
                 % (BM, BN, BK, WM, WN))
    parts.append("constexpr int NWARPS_M = %d, NWARPS_N = %d, NWARPS = %d;"
                 % (nwarps_m, nwarps_n, nwarps))
    parts.append("constexpr int NTHREADS = %d;\n" % nthreads)
    parts.append("""__global__ void mm_kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          __nv_bfloat16* __restrict__ C, int n) {
    __shared__ maca_bfloat16 sA[BM][BK + 8];
    __shared__ maca_bfloat16 sB[BK][BN + 8];
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
""")
    parts.append("        #pragma unroll\n        for (int rep = 0; rep < %d; rep++) {\n"
                 % a_loads)
    parts.append("""            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BK, c = idx % BK;
            int gr = rowBase + r, gc = kt * BK + c;
            if (r < BM && idx < BM * BK) {
                const __nv_bfloat16* ptr = &A[(size_t)gr * n + gc];
                __nv_bfloat16 h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = __float2bfloat16(0.0f);
                if (!(gr < n && gc + 1 < n)) h1 = __float2bfloat16(0.0f);
                if (!(gr < n && gc + 2 < n)) h2 = __float2bfloat16(0.0f);
                if (!(gr < n && gc + 3 < n)) h3 = __float2bfloat16(0.0f);
                sA[r][c + 0] = h0; sA[r][c + 1] = h1;
                sA[r][c + 2] = h2; sA[r][c + 3] = h3;
            }
        }
""")
    parts.append("        #pragma unroll\n        for (int rep = 0; rep < %d; rep++) {\n"
                 % b_loads)
    parts.append("""            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BN, c = idx % BN;
            int gr = kt * BK + r, gc = colBase + c;
            if (r < BK && idx < BK * BN) {
                const __nv_bfloat16* ptr = &B[gr * n + gc];
                __nv_bfloat16 h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr < n && gc + 0 < n)) h0 = __float2bfloat16(0.0f);
                if (!(gr < n && gc + 1 < n)) h1 = __float2bfloat16(0.0f);
                if (!(gr < n && gc + 2 < n)) h2 = __float2bfloat16(0.0f);
                if (!(gr < n && gc + 3 < n)) h3 = __float2bfloat16(0.0f);
                sB[r][c + 0] = h0; sB[r][c + 1] = h1;
                sB[r][c + 2] = h2; sB[r][c + 3] = h3;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, maca_bfloat16, wmma::row_major> a[WM];
            #pragma unroll
            for (int i = 0; i < WM; i++)
                wmma::load_matrix_sync(a[i],
                    &sA[warpM * (16 * WM) + i * 16][kk], BK + 8);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, maca_bfloat16, wmma::row_major> b[WN];
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
    // bf16 output: each warp owns acc[i][j] uniquely (warpM/warpN select it),
    // so every lane writes its own 4 elements with no overlap.
    #pragma unroll
    for (int i = 0; i < WM; i++) {
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {
            int gc = colBase + warpN * (16 * WN) + j * 16;
            if (gr + 15 < n && gc + 15 < n) {
                unsigned row = (__lane_id() >> 4) << 2;
                unsigned col = __lane_id() & 0xf;
                #pragma unroll
                for (int t = 0; t < 4; t++)
                    C[(size_t)(gr + row + t) * n + gc + col] =
                        __float2bfloat16(acc[i][j].x[t]);
            }
        }
    }
}

torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B) {
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()), n);
    return C;
}
""")
    return "".join(parts)


def valid_configs():
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            for BK in (16, 32):
                smem = (BM * (BK + 8) + BK * (BN + 8)) * 2 + 16 * 24 * 2
                if smem > 65536:
                    continue
                for WM in (1, 2, 4):
                    for WN in (1, 2, 4):
                        if BM % (16 * WM) or BN % (16 * WN):
                            continue
                        nthreads = 64 * (BM // (16 * WM)) * (BN // (16 * WN))
                        if nthreads < 64 or nthreads > 1024:
                            continue
                        yield (BM, BN, BK, WM, WN)


def main():
    torch.cuda.set_device(0)
    n = 4096
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B
    Ab, Bb = A.bfloat16(), B.bfloat16()
    ref_b = Ab.float() @ Bb.float()

    e, _, _ = time_kernel_cold(lambda a, b: a @ b, [Ab, Bb], num_trials=25)
    eager_us = e * 1000
    print(f"[baseline] eager bf16: {eager_us:.1f} us", flush=True)

    configs = list(valid_configs())
    print(f"[sweep] {len(configs)} bf16 configs", flush=True)
    out_path = HERE / "bf16out-results.jsonl"
    best = (eager_us, "eager")

    for i, (BM, BN, BK, WM, WN) in enumerate(configs):
        name = f"bf16out_{BM}_{BN}_{BK}_{WM}_{WN}"
        src = make_kernel(BM, BN, BK, WM, WN)
        try:
            res = build_and_test(
                src, "torch::Tensor mm_kernel_fn(torch::Tensor A, torch::Tensor B);",
                name, Ab, Bb, ref_b, str(HERE / "bf16outbuild" / name),
                20, rel_tol=5e-3)
        except Exception as ex:
            res = {"name": name, "error": f"crash {type(ex).__name__}: {str(ex)[:120]}"}
        if res.get("us_mean"):
            res["speedup_vs_eager_bf16"] = round(eager_us / res["us_mean"], 3)
            if res["us_mean"] < best[0]:
                best = (res["us_mean"], name)
                print(f"[BEST] {name}: {res['us_mean']} us "
                      f"({res['speedup_vs_eager_bf16']}x)", flush=True)
        status = "OK" if res.get("correct") else "X"
        print(f"[{i+1:>3}/{len(configs)}] {name:<30} {status} "
              f"us={res.get('us_mean','-'):>7} sp={res.get('speedup_vs_eager_bf16','-'):>6}",
              flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")
        torch.cuda.empty_cache()

    print(f"[sweep] best: {best[1]} at {best[0]:.1f} us "
          f"({eager_us/best[0]:.3f}x vs eager bf16)")


if __name__ == "__main__":
    main()
