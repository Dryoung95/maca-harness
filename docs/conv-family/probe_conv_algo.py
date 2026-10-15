#!/usr/bin/env python3
"""Probe which mcdnn conv fwd algorithm x dtype combinations actually work.

fast_conv.py's first attempt queried FFT_TILING then IMPLICIT_GEMM for an
fp16-in/fp32-out conv and got MCDNN_STATUS_NOT_SUPPORTED (9) on both. That
is the same class of problem as the mcblas leading-dimension convention:
the API is a cuDNN 8.4.1 port, but which combinations the vendor actually
instantiated has to be established by brute force, not read from docs
(none available offline).

This probe tries all 8 forward algorithms x the dtype combinations that
matter, on a small conv, and reports for each:
  * the workspace query status
  * the workspace size
  * whether the execution itself succeeds
  * the max diff against an eager fp32 reference

Only the combinations that query AND execute AND match the reference are
candidates for FastConv2d.
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <mcdnn/mcdnn.h>
#include <stdexcept>
#include <string>

static mcdnnHandle_t g_handle = nullptr;
static mcdnnHandle_t handle() {
    if (!g_handle) mcdnnCreate(&g_handle);
    mcdnnSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    return g_handle;
}

const char* statname(mcdnnStatus_t s) {
    switch (s) {
        case MCDNN_STATUS_SUCCESS: return "SUCCESS";
        case MCDNN_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case MCDNN_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case MCDNN_STATUS_BAD_PARAM: return "BAD_PARAM";
        case MCDNN_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case MCDNN_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case MCDNN_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case MCDNN_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
        case MCDNN_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case MCDNN_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        default: return "OTHER";
    }
}

// dtype codes: 0=float, 1=half, 2=bfloat16
// algo: 0..7 per mcdnnConvolutionFwdAlgo_t
// math: 0=DEFAULT, 1=FMA
// Returns: [query_status, exec_status, workspace_bytes, dummy]
double g_last_maxdiff = 0.0;

std::vector<int64_t> probe(torch::Tensor x, torch::Tensor w, torch::Tensor y,
                           torch::Tensor ref,
                           int dtype_code, int algo, int math,
                           int pad_h, int pad_w, int stride_h, int stride_w,
                           int dil_h, int dil_w, int groups) {
    std::vector<int64_t> out(4, -1);

    mcdnnDataType_t dt;
    if (dtype_code == 0) dt = MCDNN_DATA_FLOAT;
    else if (dtype_code == 1) dt = MCDNN_DATA_HALF;
    else dt = MCDNN_DATA_BFLOAT16;

    mcdnnTensorDescriptor_t x_desc, y_desc;
    mcdnnFilterDescriptor_t w_desc;
    mcdnnConvolutionDescriptor_t conv_desc;
    mcdnnCreateTensorDescriptor(&x_desc);
    mcdnnCreateTensorDescriptor(&y_desc);
    mcdnnCreateFilterDescriptor(&w_desc);
    mcdnnCreateConvolutionDescriptor(&conv_desc);

    const int batch = x.size(0), in_ch = x.size(1), in_h = x.size(2), in_w = x.size(3);
    const int out_ch = w.size(0), k_h = w.size(2), k_w = w.size(3);
    const int out_h = y.size(2), out_w = y.size(3);

    mcdnnSetTensor4dDescriptor(x_desc, MCDNN_TENSOR_NCHW, dt, batch, in_ch, in_h, in_w);
    mcdnnSetFilter4dDescriptor(w_desc, dt, MCDNN_TENSOR_NCHW, out_ch, in_ch / groups, k_h, k_w);
    mcdnnSetTensor4dDescriptor(y_desc, MCDNN_TENSOR_NCHW, dt, batch, out_ch, out_h, out_w);
    mcdnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                    dil_h, dil_w, MCDNN_CROSS_CORRELATION, dt);
    mcdnnSetConvolutionMathType(conv_desc, math == 0 ? MCDNN_DEFAULT_MATH : MCDNN_FMA_MATH);
    if (groups > 1) mcdnnSetConvolutionGroupCount(conv_desc, groups);

    mcdnnConvolutionFwdAlgo_t a = (mcdnnConvolutionFwdAlgo_t)algo;
    size_t ws_bytes = 0;
    mcdnnStatus_t q = mcdnnGetConvolutionForwardWorkspaceSize(
        handle(), x_desc, w_desc, conv_desc, y_desc, a, &ws_bytes);
    out[0] = (int64_t)q;
    out[2] = (int64_t)ws_bytes;
    if (q != MCDNN_STATUS_SUCCESS) {
        mcdnnDestroyTensorDescriptor(x_desc);
        mcdnnDestroyTensorDescriptor(y_desc);
        mcdnnDestroyFilterDescriptor(w_desc);
        mcdnnDestroyConvolutionDescriptor(conv_desc);
        return out;
    }

    torch::Tensor ws;
    if (ws_bytes > 0) {
        size_t need = (ws_bytes + 3) / 4;
        ws = torch::empty({(int64_t)std::max(need, (size_t)1)},
                          torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
    }

    float alpha = 1.0f, beta = 0.0f;
    mcdnnStatus_t e = MCDNN_STATUS_SUCCESS;
    try {
        e = mcdnnConvolutionForward(handle(), &alpha,
                                    x_desc, x.data_ptr(),
                                    w_desc, w.data_ptr(),
                                    conv_desc, a,
                                    ws.defined() ? ws.data_ptr() : nullptr,
                                    ws.defined() ? (size_t)ws.numel() * 4 : 0,
                                    &beta, y_desc, y.data_ptr());
    } catch (...) { e = MCDNN_STATUS_EXECUTION_FAILED; }
    out[1] = (int64_t)e;
    if (e != MCDNN_STATUS_SUCCESS) {
        mcdnnDestroyTensorDescriptor(x_desc);
        mcdnnDestroyTensorDescriptor(y_desc);
        mcdnnDestroyFilterDescriptor(w_desc);
        mcdnnDestroyConvolutionDescriptor(conv_desc);
        return out;
    }

    // max diff against the fp32 eager reference, computed on the device
    double md = (y.to(torch::kFloat32) - ref).abs().max().item<double>();
    out[3] = 0;  // correctness decided in python against a tolerance
    mcdnnDestroyTensorDescriptor(x_desc);
    mcdnnDestroyTensorDescriptor(y_desc);
    mcdnnDestroyFilterDescriptor(w_desc);
    mcdnnDestroyConvolutionDescriptor(conv_desc);
    // stash the maxdiff via a global, simplest channel
    g_last_maxdiff = md;
    return out;
}

double last_maxdiff() { return g_last_maxdiff; }
"""

_mod = load_inline(
    name="probe_conv_algo",
    cpp_sources=["std::vector<int64_t> probe(torch::Tensor x, torch::Tensor w, torch::Tensor y, "
                 "torch::Tensor ref, int dtype_code, int algo, int math, "
                 "int pad_h, int pad_w, int stride_h, int stride_w, "
                 "int dil_h, int dil_w, int groups);"
                 "double last_maxdiff();"],
    cuda_sources=[SRC],
    functions=["probe", "last_maxdiff"],
    verbose=False,
    with_cuda=True,
)

ALGOS = ["IMPLICIT_GEMM", "IMPLICIT_PRECOMP_GEMM", "GEMM", "DIRECT",
         "FFT", "FFT_TILING", "WINOGRAD", "WINOGRAD_NONFUSED"]
DTYPES = {0: "f32", 1: "f16", 2: "bf16"}


def main():
    torch.manual_seed(0)
    in_ch, out_ch, k = 8, 16, 3
    batch, hw = 2, 32
    conv = torch.nn.Conv2d(in_ch, out_ch, k, stride=1, padding=1,
                           groups=1, bias=False).cuda()
    x32 = torch.randn(batch, in_ch, hw, hw, device="cuda")
    with torch.no_grad():
        ref = conv(x32)
    w32 = conv.weight

    results = []
    print(f"{'algo':<20} {'dtype':<6} {'math':<8} {'query':<14} {'exec':<14} "
          f"{'ws':>10}  maxdiff")
    print("-" * 88)
    for dtype_code, dname in DTYPES.items():
        if dtype_code == 0:
            x, w, y = x32, w32, torch.empty_like(ref)
        elif dtype_code == 1:
            x, w = x32.half(), w32.half()
            y = torch.empty_like(ref)  # fp32 out; mcdnn may refuse
        else:
            x, w = x32.bfloat16(), w32.bfloat16()
            y = torch.empty_like(ref)
        # also try matching-output dtype for f16/bf16
        out_variants = [y]
        if dtype_code in (1, 2):
            out_variants.append(torch.empty(ref.shape, device="cuda",
                                            dtype=x.dtype))
        for yi, yv in enumerate(out_variants):
            for algo in range(8):
                for math in (0, 1):
                    r = _mod.probe(x, w, yv, ref, dtype_code, algo, math,
                                   1, 1, 1, 1, 1, 1, 1)
                    q, e, ws, _ = r
                    md = _mod.last_maxdiff() if e == 0 else float("nan")
                    tag = f"{dname}{'/f32out' if yi == 1 else ''}"
                    if q == 0 and e == 0:
                        results.append({"algo": ALGOS[algo], "dtype": tag,
                                        "math": math, "ws": ws, "maxdiff": md})
                    qs = "ok" if q == 0 else f"st{q}"
                    es = "ok" if e == 0 else f"st{e}"
                    if q == 0:  # only print rows the query accepted
                        print(f"{ALGOS[algo]:<20} {tag:<10} {math:<8} {qs:<14} "
                              f"{es:<14} {ws:>10}  {md:.3e}" if e == 0 else
                              f"{ALGOS[algo]:<20} {tag:<10} {math:<8} {qs:<14} "
                              f"{es:<14} {ws:>10}  -")

    ok = [r for r in results if r["maxdiff"] < 1e-3]
    print("\n" + "=" * 88)
    print(f"{len(ok)}/{len(results)} query+exec combos are correct (<1e-3)")
    for r in ok:
        print(f"  {r['algo']:<20} {r['dtype']:<10} math={r['math']} "
              f"ws={r['ws']} maxdiff={r['maxdiff']:.3e}")

    out = Path(__file__).resolve().parent / "runs" / "conv_algo_probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
