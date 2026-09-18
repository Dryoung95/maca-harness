#!/usr/bin/env python3
"""
fp16 wmma matmul with shared-memory staging (v3).

Findings driving this design:
  - global int4/float4 loads and stores work correctly on C500
  - int4/float4 stores TO SHARED memory produce NaN (MACA compiler bug)
    -> shared writes must be scalar
  - fp32->fp16 conversion in registers is fine
  - wmma fragments load from shared with stride (ldm)

Pipeline per K-tile:
  1. cooperative global float4 load of A and B tiles into registers
  2. fp32->fp16 conversion, scalar store to shared (with bank padding)
  3. all warps load wmma fragments from the shared tiles (A reused across N)
  4. mma_sync
"""


def make_kernel_wmma3(BM: int, BN: int, BK: int, WM: int, WN: int):
    """
    Shared tiles: sA[BM][BK] fp16, sB[BK][BN] fp16 (+8 padding).
    Threads = 64 * NWARPS. Cooperative load covers BM*BK + BK*BN elements.
    """
    nwarps_m = BM // (16 * WM)
    nwarps_n = BN // (16 * WN)
    nwarps = nwarps_m * nwarps_n
    nthreads = 64 * nwarps
    a_elems = BM * BK
    b_elems = BK * BN
    # loads per thread for each tile, as float4 (4 floats) then scalar fp16 stores
    a_loads = (a_elems + 4 * nthreads - 1) // (4 * nthreads)
    b_loads = (b_elems + 4 * nthreads - 1) // (4 * nthreads)
    return f"""
#include <torch/extension.h>
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

__global__ void mm_wmma3_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
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
        // ---- cooperative global float4 load -> register -> fp16 -> shared (scalar)
        #pragma unroll
        for (int rep = 0; rep < {a_loads}; rep++) {{
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BK;
            int c = idx % BK;
            int gr = rowBase + r;
            int gc = kt * BK + c;
            if (r < BM && idx < BM * BK) {{
                int4 v = *reinterpret_cast<const int4*>(&A[gr * n + gc]);
                float f0 = __int_as_float(v.x), f1 = __int_as_float(v.y);
                float f2 = __int_as_float(v.z), f3 = __int_as_float(v.w);
                // bounds: clamp reads beyond n to zero
                if (!(gr < n && gc + 0 < n)) f0 = 0.0f;
                if (!(gr < n && gc + 1 < n)) f1 = 0.0f;
                if (!(gr < n && gc + 2 < n)) f2 = 0.0f;
                if (!(gr < n && gc + 3 < n)) f3 = 0.0f;
                sA[r][c + 0] = __float2half(f0);
                sA[r][c + 1] = __float2half(f1);
                sA[r][c + 2] = __float2half(f2);
                sA[r][c + 3] = __float2half(f3);
            }}
        }}
        #pragma unroll
        for (int rep = 0; rep < {b_loads}; rep++) {{
            int idx = (rep * NTHREADS + tid) * 4;
            int r = idx / BN;
            int c = idx % BN;
            int gr = kt * BK + r;
            int gc = colBase + c;
            if (r < BK && idx < BK * BN) {{
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
            }}
        }}
        __syncthreads();

        // ---- wmma compute over the K-tile
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {{
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[WM];
            #pragma unroll
            for (int i = 0; i < WM; i++) {{
                wmma::load_matrix_sync(a[i],
                    &sA[warpM * (16 * WM) + i * 16][kk], BK + 8);
            }}
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[WN];
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

    // ---- store outputs
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

torch::Tensor mm_wmma3_{BM}_{BN}_{BK}_{WM}_{WN}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTHREADS);
    mm_wmma3_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}}
"""


def wmma3_name(BM, BN, BK, WM, WN):
    return f"mm_wmma3_{BM}_{BN}_{BK}_{WM}_{WN}"
