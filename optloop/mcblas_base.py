#!/usr/bin/env python3
"""mcblas GemmEx baseline: what does the vendor-tuned GEMM achieve on C500?

The wmma C++ API line tops out at 0.68x of eager fp32 and probes show the
compute stage is 97% of runtime, so the bottleneck is mma_sync instruction
throughput rather than tiling. Before writing more custom kernels, measure
what the vendor library reaches. If mcblas is far ahead, the gap is in
instruction selection, and the next step is the raw mma intrinsic or
mctlass rather than more wmma tile tuning.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loop import time_kernel_cold  # noqa: E402

SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <mcblas.h>
#include <mc_library_types.h>

static mcblasHandle_t g_handle = nullptr;

torch::Tensor mcblas_fp32(torch::Tensor A, torch::Tensor B) {
    if (!g_handle) mcblasCreate(&g_handle);
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options());
    float alpha = 1.0f, beta = 0.0f;
    mcblasSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    mcblasGemmEx(g_handle,
        MCBLAS_OP_N, MCBLAS_OP_N, n, n, n,
        &alpha,
        B.data_ptr<float>(), MACA_R_32F, n,
        A.data_ptr<float>(), MACA_R_32F, n,
        &beta,
        C.data_ptr<float>(), MACA_R_32F, n,
        MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
    return C;
}

torch::Tensor mcblas_fp16(torch::Tensor A, torch::Tensor B) {
    if (!g_handle) mcblasCreate(&g_handle);
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options());
    float alpha = 1.0f, beta = 0.0f;
    mcblasSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    mcblasGemmEx(g_handle,
        MCBLAS_OP_N, MCBLAS_OP_N, n, n, n,
        &alpha,
        B.data_ptr<at::Half>(), MACA_R_16F, n,
        A.data_ptr<at::Half>(), MACA_R_16F, n,
        &beta,
        C.data_ptr<at::Half>(), MACA_R_16F, n,
        MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
    mcDeviceSynchronize();
    return C;
}

torch::Tensor mcblas_bf16(torch::Tensor A, torch::Tensor B) {
    if (!g_handle) mcblasCreate(&g_handle);
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options());
    float alpha = 1.0f, beta = 0.0f;
    mcblasSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    mcblasGemmEx(g_handle,
        MCBLAS_OP_N, MCBLAS_OP_N, n, n, n,
        &alpha,
        B.data_ptr<at::BFloat16>(), MACA_R_16BF, n,
        A.data_ptr<at::BFloat16>(), MACA_R_16BF, n,
        &beta,
        C.data_ptr<at::BFloat16>(), MACA_R_16BF, n,
        MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
    return C;
}

// fp16 inputs, fp32 output and accumulate
torch::Tensor mcblas_fp16_fp32out(torch::Tensor A, torch::Tensor B) {
    if (!g_handle) mcblasCreate(&g_handle);
    int n = A.size(0);
    auto C = torch::empty({n, n}, A.options().dtype(at::kFloat));
    float alpha = 1.0f, beta = 0.0f;
    mcblasSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    mcblasGemmEx(g_handle,
        MCBLAS_OP_N, MCBLAS_OP_N, n, n, n,
        &alpha,
        B.data_ptr<at::Half>(), MACA_R_16F, n,
        A.data_ptr<at::Half>(), MACA_R_16F, n,
        &beta,
        C.data_ptr<float>(), MACA_R_32F, n,
        MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
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
    Ah, Bh = A.half(), B.half()
    Ab, Bb = A.bfloat16(), B.bfloat16()

    import os
    from torch.utils.cpp_extension import load_inline
    os.makedirs("mcblasbuild", exist_ok=True)
    m = load_inline(name="mcblas_base",
                    cpp_sources=["torch::Tensor mcblas_fp32(torch::Tensor A, torch::Tensor B);"
                                 "torch::Tensor mcblas_fp16(torch::Tensor A, torch::Tensor B);"
                                 "torch::Tensor mcblas_bf16(torch::Tensor A, torch::Tensor B);"
                                 "torch::Tensor mcblas_fp16_fp32out(torch::Tensor A, torch::Tensor B);"],
                    cuda_sources=[SRC],
                    functions=["mcblas_fp32", "mcblas_fp16", "mcblas_bf16", "mcblas_fp16_fp32out"],
                    build_directory="mcblasbuild",
                    extra_ldflags=["-L/opt/maca-3.3.0/lib64", "-lmcblas"],
                    verbose=False)
    print("built", flush=True)

    cases = [
        ("eager fp32",   lambda a, b: a @ b, A, B, ref),
        ("eager fp16",   lambda a, b: a @ b, Ah, Bh, ref),
        ("mcblas fp32",  m.mcblas_fp32, A, B, ref),
        ("mcblas fp16",  m.mcblas_fp16, Ah, Bh, ref),
        ("mcblas bf16",  m.mcblas_bf16, Ab, Bb, ref),
        ("mcblas fp16->fp32", m.mcblas_fp16_fp32out, Ah, Bh, ref),
    ]
    out_path = HERE / "mcblas-results.jsonl"
    for label, fn, X, Y, r in cases:
        out = fn(X, Y)
        torch.cuda.synchronize()
        d = (out.float() - r).abs().max().item()
        ratio = d / r.abs().max().item()
        t, tm, ts = time_kernel_cold(fn, [X, Y], num_trials=25)
        rec = {"label": label, "us_mean": round(t * 1000, 1),
               "us_min": round(tm * 1000, 1), "err_ratio": f"{ratio:.2e}"}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{label:20s} {t*1000:8.1f} us  (min {tm*1000:7.1f})  err={ratio:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
