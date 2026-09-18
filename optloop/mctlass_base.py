#!/usr/bin/env python3
"""mctlass device-GEMM baseline.

wmma C++ API tops out at 0.68x of TF32 eager and 0.067x of bf16 eager, and
probes show compute is 97% of runtime — the mma_sync path itself is slow.
mctlass is MACA's CUTLASS analog with vendor-tuned kernels; this measures
whether its device-level GEMM reaches eager speed. If it does, the gap is
in instruction selection and the harness should build on mctlass rather
than hand-rolled wmma.
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loop import time_kernel_cold  # noqa: E402

SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mctlass/mctlass.h>
#include <mctlass/gemm/device/gemm.h>

// C = A @ B with row-major output. mctlass is column-major by default, so
// compute C^T = B^T @ A^T and write into the same buffer.
torch::Tensor mctlass_fp32(torch::Tensor A, torch::Tensor B) {
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options());
    // A row-major / B column-major is the DefaultMmaCore combination mctlass actually
    // specializes for fp32 SIMT; ColumnMajor output routes through a partial
    // specialization whose leading dimensions are not transposed, so it computes B@A.
    using Gemm = mctlass::gemm::device::Gemm<
        float, mctlass::layout::RowMajor,
        float, mctlass::layout::ColumnMajor,
        float, mctlass::layout::RowMajor>;
    Gemm gemm_op;
    float alpha = 1.0f, beta = 0.0f;
    Gemm::Arguments args{
        {n, n, n},
        {A.data_ptr<float>(), n},
        {B.data_ptr<float>(), n},
        {C.data_ptr<float>(), n},
        {C.data_ptr<float>(), n},
        {alpha, beta},
        1
    };
    size_t ws = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (ws > 0) cudaMalloc(&workspace, ws);
    mctlass::Status status = gemm_op.initialize(args, workspace);
    std::cerr << "init status: " << (int)status << " ws=" << ws << std::endl;
    if (status != mctlass::Status::kSuccess) { if (workspace) cudaFree(workspace); return C; }
    status = gemm_op();
    std::cerr << "run status: " << (int)status << std::endl;
    if (workspace) cudaFree(workspace);
    return C;
}
"""


def main():
    torch.cuda.set_device(0)
    n = 4096
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    ref = A @ B

    os.makedirs("mctlassbuild", exist_ok=True)
    from torch.utils.cpp_extension import load_inline
    m = load_inline(name="mctlass_base",
                    cpp_sources=["torch::Tensor mctlass_fp32(torch::Tensor A, torch::Tensor B);"],
                    cuda_sources=[SRC],
                    functions=["mctlass_fp32"],
                    build_directory="mctlassbuild",
                    extra_cuda_cflags=["-I/opt/maca-3.3.0/include",
                                       "-I/opt/maca-3.3.0/tools/cu-bridge/include",
                                       "-std=c++17",
                                       "-fno-inline",
                                       "-DMCTLASS_ENABLE_TENSOR_CORE_MMA"],
                    verbose=False)
    print("built", flush=True)

    out = m.mctlass_fp32(A, B)
    torch.cuda.synchronize()
    d = (out - ref).abs().max().item()
    print(f"correctness: max_abs={d:.4g} ratio={d/ref.abs().max().item():.3e}")

    t, tm, ts = time_kernel_cold(m.mctlass_fp32, [A, B], num_trials=20)
    print(f"mctlass fp32: {t*1000:.1f} us (min {tm*1000:.1f}, std {ts*1000:.1f})")

    e, _, _ = time_kernel_cold(lambda a, b: a @ b, [A, B], num_trials=20)
    print(f"eager fp32 (TF32): {e*1000:.1f} us -> speedup {e/t:.3f}x")


if __name__ == "__main__":
    main()
