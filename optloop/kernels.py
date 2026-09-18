#!/usr/bin/env python3
"""
Kernel template library for the C500 optimization loop.

Design principles (C500 / MACA specific):
  - warp size = 64 (not 32): bank-conflict padding uses +8 for 64-bank shared
  - 104 SM, 2048 threads/SM, 64KB shared per block, 131072 regs per SM
  - each thread computes T consecutive outputs along the N dimension to amortize
    the A-tile broadcast and reduce register pressure per element
  - cooperative A/B tile loads with exact index arithmetic (no ambiguity)
"""

# ---------------------------------------------------------------------------
# v1: correct baseline tiled kernel (16x16), ported from upstream few-shot
# ---------------------------------------------------------------------------
KERNEL_TILED16 = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void mm_tiled_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = (row < n && t * TILE + threadIdx.x < n) ? A[row * n + t * TILE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (t * TILE + threadIdx.y < n && col < n) ? B[(t * TILE + threadIdx.y) * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = acc;
}

torch::Tensor mm_tiled(torch::Tensor A, torch::Tensor B) {
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options());
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    mm_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}
"""

# ---------------------------------------------------------------------------
# v4: correct + parametrized. Block computes a BM x BN output tile.
#     Threads laid out as (BM, BN/T). Each thread computes T outputs along N.
#     Requires BM == BN (square tiles) so the cooperative loads stay in bounds.
# ---------------------------------------------------------------------------
def make_kernel_v4(BM: int, BN: int, T: int):
    """Correct parametrized tiled matmul. threads = (BM, BN/T). Requires BM==BN."""
    assert BM == BN, "v4 requires square tiles (BM==BN) for in-bounds cooperative loads"
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = {BM};
constexpr int BN = {BN};
constexpr int T = {T};
constexpr int NT = BN / T;   // threads along x

__global__ void mm_v4_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int n) {{
    __shared__ float sA[BM][BN];
    __shared__ float sB[BN][BN];

    const int ty = threadIdx.y;          // 0..BM-1
    const int tx = threadIdx.x;          // 0..NT-1
    const int row = blockIdx.y * BM + ty;
    const int colBase = blockIdx.x * BN + tx * T;

    float acc[T];
    #pragma unroll
    for (int i = 0; i < T; i++) acc[i] = 0.0f;

    const int numKTiles = (n + BN - 1) / BN;
    for (int kt = 0; kt < numKTiles; kt++) {{
        // ---- load A tile: BM rows x BN cols (thread (ty,tx) -> T cols)
        #pragma unroll
        for (int rep = 0; rep < T; rep++) {{
            int aCol = kt * BN + tx + rep * NT;
            float v = 0.0f;
            if (row < n && aCol < n) v = A[row * n + aCol];
            sA[ty][tx + rep * NT] = v;
        }}
        // ---- load B tile: BN rows (K dim) x BN cols (N dim)
        //     thread (ty,tx) owns row kt*BN+ty, T columns starting at colBase
        #pragma unroll
        for (int tt = 0; tt < T; tt++) {{
            int bRow = kt * BN + ty;
            int bCol = colBase + tt;
            float v = 0.0f;
            if (bRow < n && bCol < n) v = B[bRow * n + bCol];
            sB[ty][tx * T + tt] = v;
        }}
        __syncthreads();

        // ---- compute
        #pragma unroll
        for (int i = 0; i < BN; i++) {{
            float a = sA[ty][i];
            #pragma unroll
            for (int tt = 0; tt < T; tt++) {{
                acc[tt] += a * sB[i][tx * T + tt];
            }}
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int tt = 0; tt < T; tt++) {{
        int c = colBase + tt;
        if (row < n && c < n) C[row * n + c] = acc[tt];
    }}
}}

torch::Tensor mm_v4_{BM}_{BN}_{T}(torch::Tensor A, torch::Tensor B) {{
    int n = A.size(0);
    auto C = torch::empty({{n, n}}, A.options());
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block(NT, BM);
    mm_v4_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
    return C;
}}
"""

# ---------------------------------------------------------------------------
# v5: v4 + bank-conflict padding on shared tiles (warp 64 -> pad 8 floats)
# ---------------------------------------------------------------------------
def make_kernel_v5(BM: int, BN: int, T: int):
    src = make_kernel_v4(BM, BN, T)
    # pad shared arrays
    src = src.replace("__shared__ float sA[BM][BN];",
                      "__shared__ float sA[BM][BN + 8];")
    src = src.replace("__shared__ float sB[BN][BN];",
                      "__shared__ float sB[BN][BN + 8];")
    # rename kernel + function to v5
    src = src.replace("mm_v4_kernel", "mm_v5_kernel")
    src = src.replace(f"mm_v4_{BM}_{BN}_{T}", f"mm_v5_{BM}_{BN}_{T}")
    return src


# ---------------------------------------------------------------------------
# v6: v5 + manual K-dimension unrolling by U (accumulate U inner products
#     per iteration to expose ILP to the compiler)
# ---------------------------------------------------------------------------
def make_kernel_v6(BM: int, BN: int, T: int, U: int):
    src = make_kernel_v5(BM, BN, T)
    inner = """        #pragma unroll
        for (int i = 0; i < BN; i++) {
            float a = sA[ty][i];
            #pragma unroll
            for (int tt = 0; tt < T; tt++) {
                acc[tt] += a * sB[i][tx * T + tt];
            }
        }"""
    inner_u = f"""        #pragma unroll
        for (int i = 0; i < BN; i += {U}) {{
            #pragma unroll
            for (int u = 0; u < {U}; u++) {{
                float a = sA[ty][i + u];
                #pragma unroll
                for (int tt = 0; tt < T; tt++) {{
                    acc[tt] += a * sB[i + u][tx * T + tt];
                }}
            }}
        }}"""
    src = src.replace(inner, inner_u)
    src = src.replace("mm_v5_kernel", "mm_v6_kernel")
    src = src.replace(f"mm_v5_{BM}_{BN}_{T}", f"mm_v6_{BM}_{BN}_{T}_{U}")
    return src


def cpp_name(kind: str, BM: int, BN: int, T: int, U: int = 0):
    if kind == "v4":
        return f"mm_v4_{BM}_{BN}_{T}"
    if kind == "v5":
        return f"mm_v5_{BM}_{BN}_{T}"
    if kind == "v6":
        return f"mm_v6_{BM}_{BN}_{T}_{U}"
    raise ValueError(kind)


def make_kernel(kind: str, BM: int, BN: int, T: int, U: int = 0):
    if kind == "v4":
        return make_kernel_v4(BM, BN, T)
    if kind == "v5":
        return make_kernel_v5(BM, BN, T)
    if kind == "v6":
        return make_kernel_v6(BM, BN, T, U)
    raise ValueError(kind)
