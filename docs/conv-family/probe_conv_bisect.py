#!/usr/bin/env python3
"""Bisect why mcdnnConvolutionForward disagrees with torch conv.

probe_conv_algo.py got maxdiff 3.7 even for f32/f32 + FFT_TILING + FMA_MATH
— a config structurally identical to the vendor conv_fprop sample, which
passes at 1.9e-07 against its own CPU reference. So mcdnn is not broken
wholesale; something in my configuration differs from the vendor's in a
way that matters.

This probe holds everything else fixed and changes ONE variable at a time
from the vendor's known-good config:

  vendor: batch=1, in_ch=4, 16x16, out_ch=16, k=5, pad=1, alpha=2.5/beta=4
  mine:   batch=2, in_ch=8, 32x32, out_ch=16, k=3, pad=1, alpha=1/beta=0

Each step applies one change and compares against torch.nn.functional.conv2d
(cross-correlation, same as the vendor CPU ref). The first step that breaks
is the guilty variable.

alpha/beta are always 1/0 here: the vendor's 2.5/4.0 multiply a
pre-initialised y, which is a different computation, not a different
convolution.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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

// mode: 0 = CROSS_CORRELATION, 1 = CONVOLUTION
// algo: index into mcdnnConvolutionFwdAlgo_t
// math: 0 = DEFAULT, 1 = FMA
// Returns the raw mcdnn status of the forward call (0 = success).
int conv_f32(torch::Tensor x, torch::Tensor w, torch::Tensor y,
             int pad_h, int pad_w, int stride_h, int stride_w,
             int dil_h, int dil_w, int groups,
             int mode, int algo, int math) {
    const int batch = x.size(0), in_ch = x.size(1), in_h = x.size(2), in_w = x.size(3);
    const int out_ch = w.size(0), k_h = w.size(2), k_w = w.size(3);
    const int out_h = y.size(2), out_w = y.size(3);

    mcdnnTensorDescriptor_t x_desc, y_desc;
    mcdnnFilterDescriptor_t w_desc;
    mcdnnConvolutionDescriptor_t conv_desc;
    mcdnnCreateTensorDescriptor(&x_desc);
    mcdnnCreateTensorDescriptor(&y_desc);
    mcdnnCreateFilterDescriptor(&w_desc);
    mcdnnCreateConvolutionDescriptor(&conv_desc);

    mcdnnSetTensor4dDescriptor(x_desc, MCDNN_TENSOR_NCHW, MCDNN_DATA_FLOAT,
                               batch, in_ch, in_h, in_w);
    mcdnnSetFilter4dDescriptor(w_desc, MCDNN_DATA_FLOAT, MCDNN_TENSOR_NCHW,
                               out_ch, in_ch / groups, k_h, k_w);
    mcdnnSetTensor4dDescriptor(y_desc, MCDNN_TENSOR_NCHW, MCDNN_DATA_FLOAT,
                               batch, out_ch, out_h, out_w);
    mcdnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                    dil_h, dil_w,
                                    mode == 0 ? MCDNN_CROSS_CORRELATION : MCDNN_CONVOLUTION,
                                    MCDNN_DATA_FLOAT);
    mcdnnSetConvolutionMathType(conv_desc, math == 0 ? MCDNN_DEFAULT_MATH : MCDNN_FMA_MATH);
    if (groups > 1) mcdnnSetConvolutionGroupCount(conv_desc, groups);

    mcdnnConvolutionFwdAlgo_t a = (mcdnnConvolutionFwdAlgo_t)algo;
    size_t ws_bytes = 0;
    mcdnnStatus_t q = mcdnnGetConvolutionForwardWorkspaceSize(
        handle(), x_desc, w_desc, conv_desc, y_desc, a, &ws_bytes);
    if (q != MCDNN_STATUS_SUCCESS) {
        mcdnnDestroyTensorDescriptor(x_desc);
        mcdnnDestroyTensorDescriptor(y_desc);
        mcdnnDestroyFilterDescriptor(w_desc);
        mcdnnDestroyConvolutionDescriptor(conv_desc);
        return 1000 + (int)q;  // query failure
    }

    torch::Tensor ws;
    if (ws_bytes > 0) {
        size_t need = (ws_bytes + 3) / 4;
        ws = torch::empty({(int64_t)std::max(need, (size_t)1)},
                          torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
    }

    float alpha = 1.0f, beta = 0.0f;
    mcdnnStatus_t e = mcdnnConvolutionForward(handle(), &alpha,
                                              x_desc, x.data_ptr<float>(),
                                              w_desc, w.data_ptr<float>(),
                                              conv_desc, a,
                                              ws.defined() ? ws.data_ptr() : nullptr,
                                              ws.defined() ? (size_t)ws.numel() * 4 : 0,
                                              &beta, y_desc, y.data_ptr<float>());

    mcdnnDestroyTensorDescriptor(x_desc);
    mcdnnDestroyTensorDescriptor(y_desc);
    mcdnnDestroyFilterDescriptor(w_desc);
    mcdnnDestroyConvolutionDescriptor(conv_desc);
    return (int)e;
}
"""

_mod = load_inline(
    name="probe_conv_bisect",
    cpp_sources=["int conv_f32(torch::Tensor x, torch::Tensor w, torch::Tensor y, "
                 "int pad_h, int pad_w, int stride_h, int stride_w, "
                 "int dil_h, int dil_w, int groups, int mode, int algo, int math);"],
    cuda_sources=[SRC],
    functions=["conv_f32"],
    verbose=False,
    with_cuda=True,
)

ALGOS = ["IMPLICIT_GEMM", "IMPLICIT_PRECOMP_GEMM", "GEMM", "DIRECT",
         "FFT", "FFT_TILING", "WINOGRAD", "WINOGRAD_NONFUSED"]


def run(tag, batch, in_ch, hw, out_ch, k, algo=5, math=1, mode=0,
        stride=1, pad=None, dil=1, groups=1):
    torch.manual_seed(0)
    pad = k // 2 if pad is None else pad
    x = torch.randn(batch, in_ch, hw, hw, device="cuda")
    w = torch.randn(out_ch, in_ch // groups, k, k, device="cuda")
    out_h = (hw + 2 * pad - dil * (k - 1) - 1) // stride + 1
    y = torch.zeros(batch, out_ch, out_h, out_h, device="cuda")

    status = _mod.conv_f32(x, w, y, pad, pad, stride, stride, dil, dil,
                           groups, mode, algo, math)
    with torch.no_grad():
        ref = F.conv2d(x, w, stride=stride, padding=pad, dilation=dil,
                       groups=groups)
    if status != 0:
        print(f"  {tag:<34} status={status}  (query/execute failed)")
        return {"tag": tag, "status": status, "maxdiff": None}
    md = (y - ref).abs().max().item()
    ok = "OK " if md < 1e-3 else "BAD"
    print(f"  {ok} {tag:<34} maxdiff={md:.3e}  shape={tuple(ref.shape)}")
    return {"tag": tag, "status": status, "maxdiff": md}


def main():
    print("vendor-known-good baseline, then one change at a time "
          "(algo=FFT_TILING, math=FMA, mode=CORRELATION):\n")
    steps = [
        ("A vendor exact (b1 c4 16 k5)",        dict(batch=1, in_ch=4,  hw=16, out_ch=16, k=5)),
        ("B k5->k3",                            dict(batch=1, in_ch=4,  hw=16, out_ch=16, k=3)),
        ("C batch 1->2",                        dict(batch=2, in_ch=4,  hw=16, out_ch=16, k=5)),
        ("D in_ch 4->8",                        dict(batch=1, in_ch=8,  hw=16, out_ch=16, k=5)),
        ("E out_ch 16->6",                      dict(batch=1, in_ch=4,  hw=16, out_ch=6,  k=5)),
        ("F hw 16->32",                         dict(batch=1, in_ch=4,  hw=32, out_ch=16, k=5)),
        ("G my full config (b2 c8 32 k3)",      dict(batch=2, in_ch=8,  hw=32, out_ch=16, k=3)),
    ]
    results = []
    for tag, kw in steps:
        results.append(run(tag, **kw))

    print("\n--- algorithm sweep on the vendor-exact shape ---\n")
    for algo in range(8):
        run(f"algo {ALGOS[algo]}", batch=1, in_ch=4, hw=16, out_ch=16, k=5,
            algo=algo, math=1)

    print("\n--- math sweep on the vendor-exact shape ---\n")
    for math in (0, 1):
        run(f"math {'DEFAULT' if math == 0 else 'FMA'}", batch=1, in_ch=4,
            hw=16, out_ch=16, k=5, algo=2, math=math)

    out = Path(__file__).resolve().parent / "runs" / "conv_bisect.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
