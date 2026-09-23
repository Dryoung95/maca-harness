import sys, os, torch
sys.path.insert(0, '/data/cuda-harness-migration/optloop')
from torch.utils.cpp_extension import load_inline

# Isolate each wmma primitive independently:
#  A) load A from identity tile, mma with B fragment -> should equal B
#  B) load B from identity tile, mma with A fragment -> should equal A
#  C) load both from identity -> result is 0 (identity@identity = identity)
#  D) fill A/B by hand, mma, store -> hand-verified reference path
SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template<int LDM>
__global__ void iso_kernel(const __half* __restrict__ A,
                           const __half* __restrict__ B,
                           float* __restrict__ C, int n, int mode) {
    __shared__ __half sA[16][LDM];
    __shared__ __half sB[16][LDM];
    const int tid = threadIdx.x;
    if (tid < 256) {
        int r = tid / 16, c = tid % 16;
        // A = identity, B = r*16+c (position marker)
        sA[r][c] = __float2half(r == c ? 1.0f : 0.0f);
        sB[r][c] = __float2half((float)tid);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, -7.0f);

    if (mode == 0) {              // load A only, B by hand
        wmma::load_matrix_sync(a, &sA[0][0], LDM);
        wmma::fill_fragment(b, __float2half((float)tid));
        wmma::mma_sync(acc, a, b, acc);
    } else if (mode == 1) {       // load B only, A by hand
        wmma::load_matrix_sync(b, &sB[0][0], LDM);
        wmma::fill_fragment(a, __float2half((float)tid));
        wmma::mma_sync(acc, a, b, acc);
    } else {                      // load both
        wmma::load_matrix_sync(a, &sA[0][0], LDM);
        wmma::load_matrix_sync(b, &sB[0][0], LDM);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(C, acc, (unsigned)n, wmma::mem_row_major);
}

torch::Tensor iso_test(torch::Tensor C, int mode) {
    int n = C.size(0);
    iso_kernel<16><<<1, 64>>>(nullptr, nullptr, C.data_ptr<float>(), n, mode);
    return C;
}
"""

os.makedirs('/data/cuda-harness-migration/optloop/diagbuild/iso', exist_ok=True)
m = load_inline(name='diag_iso',
    cpp_sources=["torch::Tensor iso_test(torch::Tensor C, int mode);"],
    cuda_sources=[SRC], functions=['iso_test'],
    build_directory='/data/cuda-harness-migration/optloop/diagbuild/iso', verbose=False)

n = 16
for mode, label in [(0, 'load A (identity) + hand B   -> expect each row = tid of that row'),
                    (1, 'hand A + load B (position)   -> expect value r*16+c at [r,c]'),
                    (2, 'load both (identity@marker)  -> expect marker r*16+c')]:
    C = torch.full((n, n), -7.0, device='cuda')
    out = m.iso_test(C, mode)
    torch.cuda.synchronize()
    print(f'--- {label}')
    print('   nonzero:', (out != 0).sum().item(), ' min:', out.min().item(), ' max:', out.max().item())
    print('   row0:', out[0, :6].cpu().numpy().round(1))
    print('   diag:', out.diagonal()[:6].cpu().numpy().round(1))
