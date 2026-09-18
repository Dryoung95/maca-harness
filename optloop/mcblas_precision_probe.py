#!/usr/bin/env python3
"""Precision probe for the mcblas L1-P1 submission.

Questions:
  1. Is fp16-input/fp32-accumulate safe under eval's allclose(1e-4) on the
     actual N=4096 torch.rand input, and on harder distributions?
  2. Does the fp32-input mcblas path (exact same output type as reference,
     no precision compromise at all) also beat eager?
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from torch.utils.cpp_extension import load_inline  # noqa: E402

os.makedirs(HERE / "probebuild2", exist_ok=True)

SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <mcblas.h>
#include <mc_library_types.h>

static mcblasHandle_t g_handle = nullptr;

torch::Tensor mcblas_fp32(torch::Tensor A, torch::Tensor B) {
    if (!g_handle) mcblasCreate(&g_handle);
    int n = A.size(0), m = A.size(1), k = B.size(1);
    auto C = torch::empty({n, k}, A.options());
    float alpha = 1.0f, beta = 0.0f;
    mcblasSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    mcblasGemmEx(g_handle, MCBLAS_OP_N, MCBLAS_OP_N, m, n, k, &alpha,
        B.data_ptr<float>(), MACA_R_32F, m,
        A.data_ptr<float>(), MACA_R_32F, m,
        &beta, C.data_ptr<float>(), MACA_R_32F, m,
        MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
    return C;
}

torch::Tensor mcblas_fp16acc(torch::Tensor A, torch::Tensor B) {
    if (!g_handle) mcblasCreate(&g_handle);
    int n = A.size(0), m = A.size(1), k = B.size(1);
    auto C = torch::empty({n, k}, A.options().dtype(at::kFloat));
    float alpha = 1.0f, beta = 0.0f;
    mcblasSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    mcblasGemmEx(g_handle, MCBLAS_OP_N, MCBLAS_OP_N, m, n, k, &alpha,
        B.data_ptr<at::Half>(), MACA_R_16F, m,
        A.data_ptr<at::Half>(), MACA_R_16F, m,
        &beta, C.data_ptr<float>(), MACA_R_32F, m,
        MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
    return C;
}
'''

m = load_inline(
    name="mcblas_probe",
    cpp_sources=[
        "torch::Tensor mcblas_fp32(torch::Tensor A, torch::Tensor B);",
        "torch::Tensor mcblas_fp16acc(torch::Tensor A, torch::Tensor B);"],
    cuda_sources=[SRC],
    functions=["mcblas_fp32", "mcblas_fp16acc"],
    build_directory=str(HERE / "probebuild2"),
    extra_include_paths=["/opt/maca-3.3.0/include"],
    extra_ldflags=["-L/opt/maca-3.3.0/lib64", "-lmcblas"],
    verbose=False,
)
print("built", flush=True)
import os  # noqa: E402

torch.manual_seed(42)
n = 4096

cases = {
    "rand [0,1)   (P1 actual)": torch.rand(n, n, device="cuda"),
    "randn        (signed)":    torch.randn(n, n, device="cuda"),
    "randn*10     (large mag)": torch.randn(n, n, device="cuda") * 10,
    "randn*100    (fp16 edge)": torch.randn(n, n, device="cuda") * 100,
}

print(f"\n{'case':26s} {'err(fp16acc)':>13s} {'allclose1e-4':>13s} {'err(fp32)':>11s}")
for name, dist in cases.items():
    A, B = dist.clone(), dist.clone()
    ref = A @ B
    out16 = m.mcblas_fp16acc(A.half(), B.half())
    out32 = m.mcblas_fp32(A, B)
    scale = ref.abs().max().item()
    e16 = (out16 - ref).abs().max().item() / scale
    e32 = (out32 - ref).abs().max().item() / scale
    ok = torch.allclose(ref, out16, atol=1e-4, rtol=1e-4)
    print(f"{name:26s} {e16:13.3e} {str(ok):>13s} {e32:11.3e}")

print("\ntiming (n=4096, 25 trials):")
for label, fn, X, Y in [
    ("eager fp32", lambda a, b: a @ b, None, None),
    ("mcblas fp16acc", m.mcblas_fp16acc, None, None),
    ("mcblas fp32", m.mcblas_fp32, None, None),
]:
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    if label.startswith("mcblas fp16"):
        A, B = A.half(), B.half()
    for _ in range(5):
        fn(A, B)
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ts = []
    for _ in range(25):
        ev[0].record()
        out = fn(A, B)
        ev[1].record()
        torch.cuda.synchronize()
        ts.append(ev[0].elapsed_time(ev[1]) * 1000)
    out_ref = A.float() @ B.float()
    err = (out.float() - out_ref).abs().max().item() / out_ref.abs().max().item()
    ts.sort()
    print(f"  {label:16s} {sum(ts)/len(ts):8.1f} us  (min {ts[0]:7.1f})  err={err:.2e}")
