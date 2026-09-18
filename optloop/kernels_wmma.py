#!/usr/bin/env python3
"""
fp16 wmma (tensor core) matmul for C500.

C500 via cu-bridge supports nvcuda::wmma with __half inputs and float
accumulators (16x16x16 fragments). fp32/tf32 fragments are NOT supported
(mxmaca::wmma only instantiates the __half variant).

Strategy: accumulate in fp32 via wmma, with fp16 shared tiles. Input tensors
are fp32; we convert to fp16 on load. This gives tensor-core throughput with
fp32-stable accumulation.
"""


def make_kernel_wmma(BM: int, BN: int, BK: int, WM: int = 4, DB: int = 0):
    """
    Block computes BM x BN fp32 outputs using wmma 16x16x16 tiles.
    Threads: one warp (64 threads) per WM wmma tiles along M; warps along N.
    Standard layout: each warp computes a (16*WM) x 16 output block via WM
    fragments stacked along M.
    """
    nwarps_n = BN // 16
    nwarps_m = BM // (16 * WM) if (16 * WM) <= BM else 1
    nthreads = 64 * nwarps_n * nwarps_m
    nbuf = 2 if DB else 1
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int BK = {BK};
constexpr int WM = {WM};
constexpr int NWARPS_N = {nwarps_n};
constexpr int NWARPS_M = {nwarps_m};
constexpr int DB = {DB};

__global__ void mm_wmma_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int n) {{
    // fp16 shared tiles (+8 padding for bank conflicts on warp 64)
    __shared__ __half sA[{nbuf}][BM][BK + 8];
    __shared__ __half sB[{nbuf}][BK][BN + 8];

    const int lane = threadIdx.x & 63;
    const int warpId = threadIdx.x >> 6;
    const int warpN = warpId % NWARPS_N;
    const int warpM = warpId / NWARPS_N;

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // each warp computes WM fragments of 16 rows x 16 cols
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM];
    #pragma unroll
    for (int i = 0; i < WM; i++)
        wmma::fill_fragment(acc[i], 0.0f);

    const int numKTiles = (n + BK - 1) / BK;

    for (int kt = 0; kt < numKTiles; kt++) {{
        int buf = (DB && (kt & 1)) ? 1 : 0;

        // ---- load A tile to fp16 shared (scalar stores; int4 broken on MACA)
        // 256 threads -> BM rows; each thread loads BK/(threads per row) cols
        // Simple mapping: thread t loads row t%BM, col stride
        const int tid = threadIdx.x;
        #pragma unroll
        for (int rep = 0; rep < (BM * BK + 255) / 256; rep++) {{
            int idx = rep * 256 + tid;
            int r = idx / BK;
            int c = idx % BK;
            if (r < BM) {{
                int gr = rowBase + r;
                int gc = kt * BK + c;
                sA[buf][r][c] = (gr < n && gc < n) ? __float2half(A[gr * n + gc]) : __float2half(0.0f);
            }}
        }}
        // ---- load B tile to fp16 shared
        #pragma unroll
        for (int rep = 0; rep < (BK * BN + 255) / 256; rep++) {{
            int idx = rep * 256 + tid;
            int r = idx / BN;
            int c = idx % BN;
            if (r < BK) {{
                int gr = kt * BK + r;
                int gc = colBase + c;
                sB[buf][r][c] = (gr < n && gc < n) ? __float2half(B[gr * n + gc]) : __float2half(0.0f);
            }}
        }}
        __syncthreads();

        // ---- compute with wmma
        #pragma unroll
        for (int k = 0; k < BK; k += 16) {{
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[WM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[NWARPS_N > 0 ? 1 : 1];
            #pragma unroll
            for (int i = 0; i < WM; i++) {{
                wmma::load_matrix_sync(a[i],
                    &sA[buf][warpM * WM * 16 + i * 16][k], BK + 8);
            }}
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bTile;
            wmma::load_matrix_sync(bTile,
                &sB[buf][k][warpN * 16], BN + 8);
            #pragma unroll
            for (int i = 0; i < WM; i++)
                wmma::mma_sync(acc[i], a[i], bTile, acc[i]);
        }}
        __syncthreads();
    }}

    // ---- store outputs
    #pragma unroll
    for (int i = 0; i < WM; i++) {{
        int gr = rowBase + warpM * WM * 16 + i * 16;
        int gc = colBase + warpN * 16;
        if (gr + 15 < n && gc + 15 < n) {{
            wmma::store_matrix_sync(&C[gr * n + gc], acc[i], n, wmma::mem_row_major);
        }} else {{
            // fallback for edge tiles
            float tmp[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {{
                wmma::store_matrix_sync(tmp, acc[i], 16, wmma::mem_row_major);
            }}
            for (int rr = 0; rr < 16 && gr + rr < n; rr++)
                for (int cc = 0; cc < 16 && gc + cc < n; cc++)
                    C[(gr + rr) * n + gc + cc] = tmp[rr * 16 + cc];
        }}
    }}
}}

torch::Tensor mm_wmma_{BM}_{BN}_{BK}_{WM}_{DB}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(64 * NWARPS_N * NWARPS_M);
    mm_wmma_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}}
"""


def wmma_name(BM, BN, BK, WM, DB):
    return f"mm_wmma_{BM}_{BN}_{BK}_{WM}_{DB}"
