#!/usr/bin/env python3
"""
v7 kernel family: high-performance tiled matmul for C500 (warp 64, 104 SM).

Standard 2D tiling with an unambiguous load mapping:
  - Block computes BM x BN outputs. Shared tiles sA[BM][BK], sB[BK][BN].
  - Threads = (BM/RM) x (BN/RN); each thread computes an RM x RN register micro-tile.
  - Cooperative A load: thread (ty,tx) loads the K-range [tx*RK .. +RK-1] of
    its own row ty*RM (loop over RM to cover all BM rows).
  - Cooperative B load: thread (ty,tx) loads K-row ty (BK rows total, needs
    BK <= NTY*... handled by looping) across N-range [tx*RN .. +RN-1].
  - float4 vectorization on all loads where possible.

Simplicity over cleverness: correctness is verified per-config, so the mapping
must be obviously right.
"""


def make_kernel_v7(BM: int, BN: int, BK: int, RM: int, RN: int, RK: int, DB: int = 0):
    ntx = BN // RN          # threads along x
    nty = BM // RM          # threads along y
    nthreads = ntx * nty
    nbuf = 2 if DB else 1
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int BK = {BK};
constexpr int RM = {RM};
constexpr int RN = {RN};
constexpr int RK = {RK};
constexpr int NTX = {ntx};
constexpr int NTY = {nty};
constexpr int DB = {DB};

__global__ void mm_v7_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int n) {{
    __shared__ float sA[{nbuf}][BM][BK + 8];
    __shared__ float sB[{nbuf}][BK][BN + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    float acc[RM][RN];
    #pragma unroll
    for (int i = 0; i < RM; i++)
        #pragma unroll
        for (int j = 0; j < RN; j++)
            acc[i][j] = 0.0f;

    const int numKTiles = (n + BK - 1) / BK;

    // load A row tile for this thread's RM rows: K-range [tx*RK .. +RK-1]
    // each of RM rows loaded with float4s
    const int aColBase = tx * RK;

    for (int kt = 0; kt < numKTiles; kt++) {{
        int buf = (DB && (kt & 1)) ? 1 : 0;

        // ---- load A: BM rows x BK cols. Thread ty covers rows [ty*RM..ty*RM+RM-1]
        // NOTE: scalar stores only — int4/float4 reinterpret_cast stores to
        // shared memory produce NaN on MACA/cucc (verified by isolation test).
        #pragma unroll
        for (int m = 0; m < RM; m++) {{
            int aRow = rowBase + ty * RM + m;
            int kCol = kt * BK;
            #pragma unroll
            for (int kk = 0; kk < RK; kk++) {{
                int aCol = kCol + aColBase + kk;
                sA[buf][ty * RM + m][aColBase + kk] =
                    (aRow < n && aCol < n) ? A[aRow * n + aCol] : 0.0f;
            }}
        }}
        // ---- load B: BK rows x BN cols. Thread ty covers K-rows with stride NTY.
        // Each thread's N-range is [tx*RN .. +RN-1].
        #pragma unroll
        for (int bkr = 0; bkr < BK; bkr += NTY) {{
            int bRow = kt * BK + ty + bkr;
            int bColBase = colBase + tx * RN;
            #pragma unroll
            for (int j = 0; j < RN; j++) {{
                int bCol = bColBase + j;
                sB[buf][ty + bkr][tx * RN + j] =
                    (bRow < n && bCol < n) ? B[bRow * n + bCol] : 0.0f;
            }}
        }}
        __syncthreads();

        // ---- compute micro-tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {{
            float aVals[RM];
            #pragma unroll
            for (int m = 0; m < RM; m++)
                aVals[m] = sA[buf][ty * RM + m][k];
            #pragma unroll
            for (int j = 0; j < RN; j++) {{
                float b = sB[buf][k][tx * RN + j];
                #pragma unroll
                for (int m = 0; m < RM; m++)
                    acc[m][j] += aVals[m] * b;
            }}
        }}
        __syncthreads();
    }}

    // ---- write outputs
    #pragma unroll
    for (int m = 0; m < RM; m++) {{
        int r = rowBase + ty * RM + m;
        if (r < n) {{
            #pragma unroll
            for (int j = 0; j < RN; j++) {{
                int c = colBase + tx * RN + j;
                if (c < n) C[r * n + c] = acc[m][j];
            }}
        }}
    }}
}}

torch::Tensor mm_v7_{BM}_{BN}_{BK}_{RM}_{RN}_{RK}_{DB}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NTX, NTY);
    mm_v7_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}}
"""


def v7_name(BM, BN, BK, RM, RN, RK, DB):
    return f"mm_v7_{BM}_{BN}_{BK}_{RM}_{RN}_{RK}_{DB}"
