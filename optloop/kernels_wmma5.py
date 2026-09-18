#!/usr/bin/env python3
"""
fp16 wmma matmul with a hand-written accumulator store (v5).

Background — why this exists
----------------------------
MACA 3.3.0's nvcuda::wmma::store_matrix_sync is a no-op at runtime. The source
in __clang_maca_mma_functions.h looks correct (it indexes f.x[0..3] into
p[row*ldm+col] and friends), and the identical code written by hand fills all
256 elements of a 16x16 tile, but calling the library inline fills zero.
Isolated repro: fragment filled with 1000+i, store with ldm=16, result stayed
at its fill value everywhere. So the store is dropped entirely.

This kernel keeps load_matrix_sync / mma_sync (those work — loads were verified
element-by-element against an identity A, and the mma intrinsic itself is
__builtin_mxc_mma_16x16x16f16) and replaces only the store with an explicit
per-lane write, matching the documented lane mapping:

    row = (lane >> 4) << 2        col = lane & 0xf
    acc.x[i] -> C[(row + i) * ldm + col]

Everything else follows wmma4: fp16 inputs, fp16 shared tiles, no fp32->fp16
conversion in the K loop, 16x16x16 fragments, B read via col_major.
"""

def make_kernel_wmma5(BM: int, BN: int, BK: int, WM: int, WN: int):
    nwarps_m = BM // (16 * WM)
    nwarps_n = BN // (16 * WN)
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

// MACA's store_matrix_sync is a no-op at runtime in 3.3.0, so the store is
// written out. Its lane mapping is also transposed relative to the header's
// documented (row+i, col): the computed tile lands at (col, row+i) instead.
// Verified against an identity A, where the transposed store reproduces B
// exactly (max diff 0.000000) and the documented one does not.
__device__ inline void store_acc_16x16(float* C, unsigned ldm,
                                       const wmma::fragment<wmma::accumulator,
                                       16, 16, 16, float>& f) {{
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;
    C[col * ldm + (row + 0)] = f.x[0];
    C[col * ldm + (row + 1)] = f.x[1];
    C[col * ldm + (row + 2)] = f.x[2];
    C[col * ldm + (row + 3)] = f.x[3];
}}

__global__ void mm_wmma5_kernel(const __half* __restrict__ A,
                                const __half* __restrict__ B,
                                float* __restrict__ C, int n) {{
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
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[WM];
            #pragma unroll
            for (int i = 0; i < WM; i++)
                wmma::load_matrix_sync(a[i],
                    &sA[warpM * (16 * WM) + i * 16][kk], BK + 8);
            // MACA swaps the b-fragment layout tags: row_major reads down
            // columns (x[i] = B[row+i, col]), col_major reads across rows.
            // We need the column read, so the tag is row_major.
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
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int i = 0; i < WM; i++) {{
        int gr = rowBase + warpM * (16 * WM) + i * 16;
        #pragma unroll
        for (int j = 0; j < WN; j++) {{
            int gc = colBase + warpN * (16 * WN) + j * 16;
            if (gr + 15 < n && gc + 15 < n) {{
                store_acc_16x16(&C[(size_t)gc * n + gr], (unsigned)n, acc[i][j]);
            }} else {{
                // boundary path: stage the 16x16 tile in shared memory using
                // the same transposed mapping, then copy out scalar by scalar
                __shared__ float stmp[16 * 16];
                if (warpId < NWARPS) {{
                    unsigned row = (warpId >> 4) << 2;
                    unsigned col = warpId & 0xf;
                    #pragma unroll
                    for (int t = 0; t < 4; t++)
                        stmp[col * 16 + (row + t)] = acc[i][j].x[t];
                }}
                __syncthreads();
                for (int rr = 0; rr < 16 && gr + rr < n; rr++)
                    for (int cc = 0; cc < 16 && gc + cc < n; cc++)
                        C[(size_t)(gr + rr) * n + gc + cc] = stmp[cc * 16 + rr];
            }}
        }}
    }}
}}

torch::Tensor mm_wmma5_{BM}_{BN}_{BK}_{WM}_{WN}(torch::Tensor A, torch::Tensor B) {{
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == at::kHalf, "B must be fp16");
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options().dtype(at::kFloat));
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_wmma5_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), n);
    return C;
}}
"""


def wmma5_name(BM, BN, BK, WM, WN):
    return f"mm_wmma5_{BM}_{BN}_{BK}_{WM}_{WN}"
