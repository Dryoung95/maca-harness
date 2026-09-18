#!/usr/bin/env python3
"""
fp16 wmma matmul with hand-written fragment load/store (v6).

What v5/v4 got wrong, and how this was pinned down
--------------------------------------------------
MACA 3.3.0's nvcuda::wmma has two broken primitives, found by isolating each
one against an identity-A reference:

  1. store_matrix_sync is a no-op at runtime. Filling a fragment with known
     values and storing leaves the destination untouched; the identical index
     arithmetic written by hand fills all 256 elements. So the store must be
     written out.

  2. load_matrix_sync scrambles the fragment contents. Filling fragments by
     hand from the documented lane mapping, running mma_sync, and storing with
     the documented (non-transposed) store reproduces A@B to 1e-6. Using
     load_matrix_sync for the same fragments does not. So the loads must be
     written out too.

The mma intrinsic itself (__builtin_mxc_mma_16x16x16f16 via mma_sync) is
correct, as is the fragment storage layout. This kernel therefore keeps
mma_sync and replaces both load and store with the documented index math:

    a (row_major):  row = lane & 0xf,   col = (lane >> 4) << 2
                    a.x[i] = A[row, col + i]
    b (col_major):  row = (lane >> 4) << 2,  col = lane & 0xf
                    b.x[i] = B[row + i, col]
    acc store:      row = (lane >> 4) << 2,  col = lane & 0xf
                    C[(row + i) * ldm + col] = acc.x[i]

Everything else follows wmma5: fp16 inputs, fp16 shared tiles, no fp32->fp16
conversion inside the K loop, 16x16x16 fragments.
"""


def make_kernel_wmma6(BM: int, BN: int, BK: int, WM: int, WN: int):
    nwarps_m = BM // (16 * WM)
    nwarps_n = BN // (16 * WN)
    nwarps = nwarps_m * nwarps_n
    nthreads = 64 * nwarps
    a_elems = BM * BK
    b_elems = BK * BN
    # each thread moves 4 consecutive elements per iteration; the loops below
    # compute row/col from a linear element counter
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

// Hand-written equivalents of load_matrix_sync / store_matrix_sync. The MACA
// 3.3.0 library inlines for these are broken (loads scramble data, store is a
// no-op); mma_sync itself is correct, so fragments are filled and drained here.
__device__ inline void load_a_16x16(
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>& f,
        const __half* p, unsigned ldm) {{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 2;
    unsigned short* dst = reinterpret_cast<unsigned short*>(&f);
    const unsigned short* src = reinterpret_cast<const unsigned short*>(p);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        dst[i] = src[row * ldm + (col + i)];
}}

__device__ inline void load_b_16x16(
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>& f,
        const __half* p, unsigned ldm) {{
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;
    unsigned short* dst = reinterpret_cast<unsigned short*>(&f);
    const unsigned short* src = reinterpret_cast<const unsigned short*>(p);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        dst[i] = src[(row + i) * ldm + col];
}}

__device__ inline void store_acc_16x16(
        float* C, unsigned ldm,
        const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& f) {{
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        C[(row + i) * ldm + col] = f.x[i];
}}

__global__ void mm_wmma6_kernel(const __half* __restrict__ A,
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
        // sA is [BM][BK+8]: rows are M, columns are K. Thread tid moves
        // elements [tid*4 .. tid*4+3] of a linear walk over the tile.
        #pragma unroll
        for (int rep = 0; rep < {a_loads}; rep++) {{
            int base = rep * NTHREADS * 4 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++) {{
                int e = base + i;
                if (e >= BM * BK) break;
                int r = e / BK;
                int c = e % BK;
                int gr = rowBase + r;
                int gc = kt * BK + c;
                __half v = __float2half(0.0f);
                if (gr < n && gc < n) v = A[(size_t)gr * n + gc];
                sA[r][c] = v;
            }}
        }}
        // sB is [BK][BN+8]: rows are K, columns are N, walked linearly.
        #pragma unroll
        for (int rep = 0; rep < {b_loads}; rep++) {{
            int base = rep * NTHREADS * 4 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++) {{
                int e = base + i;
                if (e >= BK * BN) break;
                int r = e % BK;
                int c = e / BK;
                int gr = kt * BK + r;
                int gc = colBase + c;
                __half v = __float2half(0.0f);
                if (gr < n && gc < n) v = B[(size_t)gr * n + gc];
                sB[r][c] = v;
            }}
        }}
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {{
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[WM];
            #pragma unroll
            for (int i = 0; i < WM; i++)
                load_a_16x16(a[i],
                    &sA[warpM * (16 * WM) + i * 16][kk], BK + 8);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[WN];
            #pragma unroll
            for (int j = 0; j < WN; j++)
                load_b_16x16(b[j],
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
                store_acc_16x16(&C[(size_t)gr * n + gc], (unsigned)n, acc[i][j]);
            }} else {{
                // boundary path: stage through shared with the same mapping
                __shared__ float stmp[16 * 16];
                if (warpId < NWARPS) {{
                    unsigned row = (warpId >> 4) << 2;
                    unsigned col = warpId & 0xf;
                    #pragma unroll
                    for (int t = 0; t < 4; t++)
                        stmp[(row + t) * 16 + col] = acc[i][j].x[t];
                }}
                __syncthreads();
                for (int rr = 0; rr < 16 && gr + rr < n; rr++)
                    for (int cc = 0; cc < 16 && gc + cc < n; cc++)
                        C[(size_t)(gr + rr) * n + gc + cc] = stmp[rr * 16 + cc];
            }}
        }}
    }}
}}

torch::Tensor mm_wmma6_{BM}_{BN}_{BK}_{WM}_{WN}(torch::Tensor A, torch::Tensor B) {{
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == at::kHalf, "B must be fp16");
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options().dtype(at::kFloat));
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_wmma6_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), n);
    return C;
}}
"""


def wmma6_name(BM, BN, BK, WM, WN):
    return f"mm_wmma6_{BM}_{BN}_{BK}_{WM}_{WN}"
