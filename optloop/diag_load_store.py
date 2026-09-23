import sys, os, torch
sys.path.insert(0, '/data/cuda-harness-migration/optloop')
from torch.utils.cpp_extension import load_inline

# Pure round-trip: load a known tile into a fragment, store it back.
# No mma, no fill semantics. If load/store are correct, output == input exactly.
SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template<int LDM>
__global__ void roundtrip_kernel(const __half* __restrict__ src,
                                 float* __restrict__ C, int n) {
    __shared__ __half tile[16][LDM];
    const int tid = threadIdx.x;
    if (tid < 256) {
        int r = tid / 16, c = tid % 16;
        tile[r][c] = src == nullptr ? __float2half((float)(r * 16 + c))
                                    : src[r * n + c];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::load_matrix_sync(a, &tile[0][0], LDM);

    // store fragment back as accumulator via mma against identity-free path:
    // simplest: copy each element through the fragment's .x array
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    #pragma unroll
    for (int i = 0; i < acc.num_elements; i++) {
        acc.x[i] = (float)a.x[i];
    }
    wmma::store_matrix_sync(C, acc, (unsigned)n, wmma::mem_row_major);
}

torch::Tensor roundtrip(torch::Tensor C, int ldm) {
    int n = C.size(0);
    if (ldm == 16) roundtrip_kernel<16><<<1, 64>>>(nullptr, C.data_ptr<float>(), n);
    else           roundtrip_kernel<24><<<1, 64>>>(nullptr, C.data_ptr<float>(), n);
    return C;
}
"""

os.makedirs('/data/cuda-harness-migration/optloop/diagbuild/rt', exist_ok=True)
m = load_inline(name='diag_rt',
    cpp_sources=["torch::Tensor roundtrip(torch::Tensor C, int ldm);"],
    cuda_sources=[SRC], functions=['roundtrip'],
    build_directory='/data/cuda-harness-migration/optloop/diagbuild/rt', verbose=False)

import torch
n = 16
expected = torch.arange(n * n, device='cuda').float().reshape(n, n)
for ldm in (16, 24):
    C = torch.zeros(n, n, device='cuda')
    out = m.roundtrip(C, ldm)
    torch.cuda.synchronize()
    exact = bool(torch.equal(out, expected))
    # also report how many positions match and which pattern the mismatch takes
    match = int((out == expected).sum())
    print(f'ldm={ldm}: exact={exact}  matching={match}/256')
    if not exact:
        bad = (out != expected).nonzero()
        print('   first mismatches:', [(int(r), int(c), float(out[r, c]), float(expected[r, c]))
                                       for r, c in bad[:5]])
