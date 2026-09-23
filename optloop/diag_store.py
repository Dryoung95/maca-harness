import sys, os, torch
sys.path.insert(0,'.')
from torch.utils.cpp_extension import load_inline

# Test 1: fill fragment with known value, store with library store.
# If store is a no-op, output stays 0.
SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

__global__ void fill_store_kernel(float* C, int n) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
    // fill with row*100+col so we can see exactly where each value lands
    #pragma unroll
    for (int i = 0; i < 4; i++)
        f.x[i] = 100.0f * ((__lane_id() >> 4) * 4 + i) + (__lane_id() & 0xf);
    wmma::store_matrix_sync(C, f, (unsigned)n, wmma::mem_row_major);
}

torch::Tensor diag_store(torch::Tensor C) {
    int n = C.size(0);
    fill_store_kernel<<<1, 64>>>(C.data_ptr<float>(), n);
    return C;
}
"""
os.makedirs('diagbuild/store', exist_ok=True)
m = load_inline(name='diag_store', cpp_sources=["torch::Tensor diag_store(torch::Tensor C);"],
                cuda_sources=[SRC], functions=['diag_store'],
                build_directory='diagbuild/store', verbose=False)
C = torch.zeros(16,16, device='cuda')
out = m.diag_store(C)
torch.cuda.synchronize()
print("library store_matrix_sync output (16x16):")
print(out.cpu().numpy().astype(int))
print("nonzero count:", (out != 0).sum().item())
print("all-zero rows:", [i for i in range(16) if bool((out[i] == 0).all())])
zero_pos = (out == 0).nonzero()
print("zero element positions:", zero_pos.cpu().numpy().tolist())
if zero_pos.numel():
    r, c = zero_pos[0]
    print("expected value at that position:", int(r) * 100 + int(c))
