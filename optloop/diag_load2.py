import sys, os, torch
sys.path.insert(0,'.')
from torch.utils.cpp_extension import load_inline

# Same identity test, but with padded shared tiles (ldm = 16+8 = 24), exactly
# like wmma3 uses. If the padded version is correct, load_matrix_sync is fine
# and the earlier "scrambles data" finding was a padding/stride artifact.
SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template<int LDM>
__global__ void diag_kernel(const __half* __restrict__ A,
                            const __half* __restrict__ B,
                            float* __restrict__ C, int n) {
    __shared__ __half sA[16][LDM];
    __shared__ __half sB[16][LDM];
    const int tid = threadIdx.x;
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

    wmma::load_matrix_sync(a, &sA[0][0], LDM);
    wmma::load_matrix_sync(b, &sB[0][0], LDM);
    wmma::mma_sync(acc, a, b, acc);
    wmma::store_matrix_sync(C, acc, (unsigned)n, wmma::mem_row_major);
}

torch::Tensor diag_pad16(torch::Tensor C) {
    int n = C.size(0);
    diag_kernel<16><<<1, 64>>>(nullptr, nullptr, C.data_ptr<float>(), n);
    return C;
}
torch::Tensor diag_pad24(torch::Tensor C) {
    int n = C.size(0);
    diag_kernel<24><<<1, 64>>>(nullptr, nullptr, C.data_ptr<float>(), n);
    return C;
}
"""
os.makedirs('diagbuild/load2', exist_ok=True)
m = load_inline(name='diag_load2',
    cpp_sources=["torch::Tensor diag_pad16(torch::Tensor C); torch::Tensor diag_pad24(torch::Tensor C);"],
    cuda_sources=[SRC], functions=['diag_pad16','diag_pad24'],
    build_directory='diagbuild/load2', verbose=False)
n=16
expected = torch.arange(n*n, device='cuda').float().reshape(n,n)
for tag, fn in [("ldm=16 (unpadded)", m.diag_pad16), ("ldm=24 (padded)", m.diag_pad24)]:
    C = torch.zeros(n,n, device='cuda')
    out = fn(C); torch.cuda.synchronize()
    ok = torch.allclose(out, expected, atol=1e-3)
    print(f"{tag}: correct={ok}  first row={out[0,:4].cpu().numpy().round(2)}")
