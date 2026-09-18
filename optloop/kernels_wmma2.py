#!/usr/bin/env python3
"""
fp16 wmma matmul, global-memory fragment loads (no shared tiles).

The K-tile is consumed directly from global memory via wmma::load_matrix_sync,
which accepts a pointer + leading dimension. Input tensors are pre-converted
to fp16 once (amortized over all K-tiles), accumulation in fp32.

This eliminates the scalar fp16-conversion + shared-store bottleneck measured
in the first wmma attempt.
"""


def make_kernel_wmma2(BM: int, BN: int, WM: int, WN: int):
    """
    Block computes BM x BN outputs.
    NWARPS_M = BM / (16*WM) warps along M, NWARPS_N = BN / (16*WN) along N.
    Each warp computes WM x WN fragments of 16x16.
    """
    nwarps_m = BM // (16 * WM)
    nwarps_n = BN // (16 * WN)
    nwarps = nwarps_m * nwarps_n
    nthreads = 64 * nwarps
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int WM = {WM};
constexpr int WN = {WN};
constexpr int NWARPS_M = {nwarps_m};
constexpr int NWARPS_N = {nwarps_n};
constexpr int NWARPS = {nwarps};

__global__ void mm_wmma2_kernel(const __half* __restrict__ A,
                                const __half* __restrict__ B,
                                float* __restrict__ C, int n) {{
    const int warpId = threadIdx.x >> 6;
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

    const int numKTiles = (n + 15) / 16;

    for (int kt = 0; kt < numKTiles; kt++) {{
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[WM];
        #pragma unroll
        for (int i = 0; i < WM; i++) {{
            int aRow = rowBase + warpM * (16 * WM) + i * 16;
            wmma::load_matrix_sync(a[i], &A[aRow * n + kt * 16], n);
        }}
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[WN];
        #pragma unroll
        for (int j = 0; j < WN; j++) {{
            int bCol = colBase + warpN * (16 * WN) + j * 16;
            wmma::load_matrix_sync(b[j], &B[(kt * 16) * n + bCol], n);
        }}
        #pragma unroll
        for (int i = 0; i < WM; i++)
            #pragma unroll
            for (int j = 0; j < WN; j++)
                wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }}

    // store outputs
    #pragma unroll
    for (int i = 0; i < WM; i++) {{
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {{
            int gc = colBase + warpN * (16 * WN) + j * 16;
            if (gr + 15 < n && gc + 15 < n) {{
                wmma::store_matrix_sync(&C[gr * n + gc], acc[i][j], n, wmma::mem_row_major);
            }} else {{
                float tmp[16];
                wmma::store_matrix_sync(tmp, acc[i][j], 16, wmma::mem_row_major);
                for (int rr = 0; rr < 16 && gr + rr < n; rr++)
                    for (int cc = 0; cc < 16 && gc + cc < n; cc++)
                        C[(gr + rr) * n + gc + cc] = tmp[rr * 16 + cc];
            }}
        }}
    }}
}}

torch::Tensor mm_wmma2_{BM}_{BN}_{WM}_{WN}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options().dtype(torch::kFloat32));
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(64 * NWARPS);
    mm_wmma2_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), n);
    return C;
}}
"""


def wmma2_name(BM, BN, WM, WN):
    return f"mm_wmma2_{BM}_{BN}_{WM}_{WN}"
