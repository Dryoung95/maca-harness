import sys, os, torch
sys.path.insert(0,'.')
from torch.utils.cpp_extension import load_inline

# ldm=24 padded: load is correct (V2 verified to 1.6e-6 earlier). So the
# remaining question is only the store layout tag. Test mem_row_major vs
# mem_col_major with a fragment whose value encodes (row, col).
SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template<int MEM>
__global__ void diag_kernel(float* __restrict__ C, int n) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
    // value = 100*row + col, so the destination pattern reveals the mapping
    #pragma unroll
    for (int i = 0; i < 4; i++)
        f.x[i] = 100.0f * ((__lane_id() >> 4) * 4 + i) + (__lane_id() & 0xf);
    wmma::store_matrix_sync(C, f, (unsigned)n, (wmma::layout_t)MEM);
}

torch::Tensor diag_row(torch::Tensor C) { diag_kernel<16><<<1, 64>>>(C.data_ptr<float>(), C.size(0)); return C; }
torch::Tensor diag_col(torch::Tensor C) { diag_kernel<17><<<1, 64>>>(C.data_ptr<float>(), C.size(0)); return C; }
torch::Tensor diag_undef(torch::Tensor C) { diag_kernel<0><<<1, 64>>>(C.data_ptr<float>(), C.size(0)); return C; }
"""
os.makedirs('diagbuild/store2', exist_ok=True)
m = load_inline(name='diag_store2',
    cpp_sources=["torch::Tensor diag_row(torch::Tensor C); torch::Tensor diag_col(torch::Tensor C); torch::Tensor diag_undef(torch::Tensor C);"],
    cuda_sources=[SRC], functions=['diag_row','diag_col','diag_undef'],
    build_directory='diagbuild/store2', verbose=False)
n=16
for tag, fn, expected in [
    ("mem_row_major", m.diag_row, torch.arange(n*n).float().reshape(n,n)),
    ("mem_col_major", m.diag_col, torch.arange(n*n).float().reshape(n,n).T.contiguous()),
]:
    C = torch.zeros(n,n, device='cuda')
    out = fn(C); torch.cuda.synchronize()
    ok = torch.equal(out.to(torch.int64), expected.to(torch.int64).cuda())
    print(f"{tag}: matches_expected={ok}")
    if not ok:
        print("  out[0,:4] =", out[0,:4].cpu().numpy().astype(int), " expected", expected[0,:4].cpu().numpy().astype(int))
        print("  out[:,0]  =", out[:,0].cpu().numpy().astype(int), " expected", expected[:,0].cpu().numpy().astype(int))
