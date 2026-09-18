import sys, os, torch
sys.path.insert(0,'.')
from torch.utils.cpp_extension import load_inline

# Load an identity matrix into a b-fragment via load_matrix_sync, run mma into
# an accumulator that started at 1000, then store. If load works, acc = A@I = A
# so C == A (shifted). If load is broken, C stays at the fill value.
SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// 16x16 tiles, one warp, BK=16. A is identity -> acc must equal B.
__global__ void diag_kernel(const __half* __restrict__ A,
                            const __half* __restrict__ B,
                            float* __restrict__ C, int n) {
    __shared__ __half sA[16][16];
    __shared__ __half sB[16][16];
    const int tid = threadIdx.x;
    // fill tiles: sA = identity, sB = row-major 0..255
    if (tid < 256) {
        int r = tid / 16, c = tid % 16;
        sA[r][c] = __float2half(r == c ? 1.0f : 0.0f);
        sB[r][c] = __float2half((float)tid);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 1000.0f);

    wmma::load_matrix_sync(a, &sA[0][0], 16);
    wmma::load_matrix_sync(b, &sB[0][0], 16);
    wmma::mma_sync(acc, a, b, acc);
    wmma::store_matrix_sync(C, acc, (unsigned)n, wmma::mem_row_major);
}

torch::Tensor diag_run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int n = C.size(0);
    diag_kernel<<<1, 64>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), n);
    return C;
}
"""
os.makedirs('diagbuild/load', exist_ok=True)
m = load_inline(name='diag_load', cpp_sources=["torch::Tensor diag_run(torch::Tensor A, torch::Tensor B, torch::Tensor C);"],
                cuda_sources=[SRC], functions=['diag_run'],
                build_directory='diagbuild/load', verbose=False)
n=16
C = torch.zeros(n,n, device='cuda')
A = torch.eye(n, device='cuda').half()
B = torch.arange(n*n, device='cuda').float().reshape(n,n).half()
out = m.diag_run(A,B,C)
torch.cuda.synchronize()
print("C after load_matrix_sync + mma (expect B = 0..255 in row-major):")
print(out.cpu().numpy().astype(int))
print("matches B?", torch.equal(out, B.float()))
print("all still 1000 (load broken)?", bool((out==1000).all()))
