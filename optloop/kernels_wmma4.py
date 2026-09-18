#!/usr/bin/env python3
"""
fp16 wmma matmul using MACA's native 16x8x16 mma shape (v4).

Three findings drive this version:

1. MACA wmma only instantiates __half mma at 16x8x16 (not the NVIDIA
   16x16x16). The earlier wmma3 kernels declared 16x16x16 fragments; those
   only compiled because the fp32 path never instantiated a half mma.
2. C500's fp16 tensor-core ceiling is 155 TFLOPS at n=4096 (885 us), 1.80x
   the fp32 eager baseline. That is the number to beat.
3. wmma3 converted fp32->fp16 inside the K-tile loop, once per element per
   tile. Removing that conversion is the point of taking fp16 inputs.

B is stored in shared memory transposed (column-major view of a row-major
tile) so that a 16x8x16 col_major b-fragment reads a contiguous 8-wide slice.
"""


def make_kernel_wmma4(BM: int, BN: int, BK: int, WM: int, WN: int):
    nwarps_m = BM // (16 * WM)
    nwarps_n = 1   # each warp already owns WN 16-wide output tiles
    nwarps = nwarps_m * nwarps_n
    nthreads = 64 * nwarps
    a_elems = BM * BK
    b_elems = BK * BN
    a_loads = (a_elems + 4 * nthreads - 1) // (4 * nthreads)
    b_loads = (b_elems + 4 * nthreads - 1) // (4 * nthreads)
    return f"""
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int BK = {BK};
constexpr int WM = {WM};
constexpr int WN = {WN};
constexpr int NWARPS_M = {nwarps_m};
constexpr int NWARPS_N = {nwarps_n};
constexpr int NWARPS = {nwarps};
constexpr int NTHREADS = {nthreads};

__global__ void mm_wmma4_kernel(const __half* __restrict__ A,
                                const __half* __restrict__ B,
                                float* __restrict__ C, int n) {{
    // sA: BM rows x BK cols, row-major, read as 16x16 row_major a-fragments.
    // sB: BK rows x BN cols in global order, read as 16x8 col_major
    // b-fragments: element (k, j) sits at sB[k][j].
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

    for (int kt = 0; kt < numKTiles; kt++) {{
        #pragma unroll
        for (int rep = 0; rep < {a_loads}; rep++) {{
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BK;
            int c = idx % BK;
            int gr = rowBase + r;
            int gc = kt * BK + c;
            if (r < BM && idx < BM * BK) {{
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
            }}
        }}
        #pragma unroll
        for (int rep = 0; rep < {b_loads}; rep++) {{
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx % BK;
            int c = idx / BK;
            int gr = kt * BK + r;
            int gc = colBase + c;
            if (c < BN && idx < BK * BN) {{
                const __half* ptr = &B[(size_t)gr * n + gc];
                __half h0 = ptr[0], h1 = ptr[1], h2 = ptr[2], h3 = ptr[3];
                if (!(gr + 0 < n && gc < n)) h0 = __float2half(0.0f);
                if (!(gr + 1 < n && gc < n)) h1 = __float2half(0.0f);
                if (!(gr + 2 < n && gc < n)) h2 = __float2half(0.0f);
                if (!(gr + 3 < n && gc < n)) h3 = __float2half(0.0f);
                sB[r + 0][c] = h0;
                sB[r + 1][c] = h1;
                sB[r + 2][c] = h2;
                sB[r + 3][c] = h3;
            }}
        }}
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {{
            // MACA: mma_sync only accepts 16x8x16 half fragments, and the
            // b-fragment layout is fixed by its 4-element lane mapping
            // (row = (lane>>4)<<2, col = lane&7), which matches a col_major
            // read of a BK-wide slice of sB.
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[WM];
            #pragma unroll
            for (int i = 0; i < WM; i++) {{
                wmma::load_matrix_sync(a[i],
                    &sA[warpM * (16 * WM) + i * 16][kk], BK + 8);
            }}
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[WN];
            #pragma unroll
            for (int j = 0; j < WN; j++) {{
                wmma::load_matrix_sync(b[j],
                    &sB[kk][warpN * (16 * WN) + j * 16], BN + 8);
            }}
            #pragma unroll
            for (int i = 0; i < WM; i++)
                #pragma unroll
                for (int j = 0; j < WN; j++)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int i = 0; i < WM; i++) {{
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {{
            int gc = colBase + warpN * (8 * WN) + j * 8;
            if (gr + 15 < n && gc + 15 < n) {{
                wmma::store_matrix_sync(&C[(size_t)gr * n + gc], acc[i][j], n, wmma::mem_row_major);
            }} else {{
                float tmp[16 * 16];
                wmma::store_matrix_sync(tmp, acc[i][j], 16, wmma::mem_row_major);
                for (int rr = 0; rr < 16 && gr + rr < n; rr++)
                    for (int cc = 0; cc < 16 && gc + cc < n; cc++)
                        C[(size_t)(gr + rr) * n + gc + cc] = tmp[rr * 16 + cc];
            }}
        }}
    }}
}}

torch::Tensor mm_wmma4_{BM}_{BN}_{BK}_{WM}_{WN}(torch::Tensor A, torch::Tensor B) {{
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == at::kHalf, "B must be fp16");
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options().dtype(at::kFloat));
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_wmma4_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), n);
    return C;
}}
"""


def wmma4_name(BM, BN, BK, WM, WN):
    return f"mm_wmma4_{BM}_{BN}_{BK}_{WM}_{WN}"
