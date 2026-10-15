#!/usr/bin/env python3
"""Time FastConv2d's mcdnn path against the gate's actual reference.

Two findings from probe_conv_f16.py decide what this probe measures:

  1. mcdnn conv in *this* SDK has no usable fp16 path (f16->f32 returns
     BAD_PARAM, f16->f16 computes the wrong thing at 5.7e-2). The GEMM
     line's fp16 trick does not transfer to conv. So the only correct
     operand type is fp32.
  2. The one exact-right combination is PRECOMP_GEMM + FMA (maxdiff
     0.000e+00), and FFT_TILING + FMA is 43% faster but sits at 1.3e-4 --
     right on the 1e-4 tolerance line.

But timing numbers only mean something against the reference the gate
actually uses: eager fp32 **with TF32 on** (the maca-harness README §3.2
states "reference: eager fp32 with TF32 on"; C500 ships allow_tf32=True by
default). My earlier probes compared against a TF32-off reference, which is
a different -- more exact -- yardstick. Under the gate's real reference,
DEFAULT_MATH (the TF32 kernels) may agree with it *better* than FMA does,
because both sides make the same TF32 approximation. That is the decisive
question.

This probe measures on real L1 shapes (P50 256x3x224x224 k11s4p2, P55
8x64x512x1024 k3p1, P62/P63 square k3) with TF32 ON, which is the number a
candidate kernel would actually be graded against. Per-combination it
reports both correctness (against the TF32-on reference) and speedup.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <mcdnn/mcdnn.h>
#include <string>
#include <vector>

static mcdnnHandle_t g_handle = nullptr;
static mcdnnHandle_t handle() {
    if (!g_handle) mcdnnCreate(&g_handle);
    mcdnnSetStream(g_handle, c10::cuda::getCurrentCUDAStream());
    return g_handle;
}

// algo: 1=PRECOMP_GEMM, 5=FFT_TILING. math: 0=DEFAULT(TF32), 1=FMA(exact)
// Returns [status, workspace_bytes]; 0 status = success.
std::vector<int64_t> conv(torch::Tensor x, torch::Tensor w, torch::Tensor y,
                          int pad_h, int pad_w, int stride_h, int stride_w,
                          int dil_h, int dil_w, int groups,
                          int algo, int math) {
    std::vector<int64_t> out(2, -1);

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
                                    dil_h, dil_w, MCDNN_CROSS_CORRELATION,
                                    MCDNN_DATA_FLOAT);
    mcdnnSetConvolutionMathType(conv_desc, math == 0 ? MCDNN_DEFAULT_MATH : MCDNN_FMA_MATH);
    if (groups > 1) mcdnnSetConvolutionGroupCount(conv_desc, groups);

    mcdnnConvolutionFwdAlgo_t a = (mcdnnConvolutionFwdAlgo_t)algo;
    size_t ws_bytes = 0;
    mcdnnStatus_t q = mcdnnGetConvolutionForwardWorkspaceSize(
        handle(), x_desc, w_desc, conv_desc, y_desc, a, &ws_bytes);
    out[1] = (int64_t)ws_bytes;
    if (q != MCDNN_STATUS_SUCCESS) {
        mcdnnDestroyTensorDescriptor(x_desc);
        mcdnnDestroyTensorDescriptor(y_desc);
        mcdnnDestroyFilterDescriptor(w_desc);
        mcdnnDestroyConvolutionDescriptor(conv_desc);
        out[0] = (int64_t)q;
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
                                    x_desc, x.data_ptr<float>(),
                                    w_desc, w.data_ptr<float>(),
                                    conv_desc, a,
                                    ws.defined() ? ws.data_ptr() : nullptr,
                                    ws.defined() ? (size_t)ws.numel() * 4 : 0,
                                    &beta, y_desc, y.data_ptr<float>());
    } catch (...) { e = MCDNN_STATUS_EXECUTION_FAILED; }
    out[0] = (int64_t)e;

    mcdnnDestroyTensorDescriptor(x_desc);
    mcdnnDestroyTensorDescriptor(y_desc);
    mcdnnDestroyFilterDescriptor(w_desc);
    mcdnnDestroyConvolutionDescriptor(conv_desc);
    return out;
}
"""

_mod = load_inline(
    name="probe_conv_gate_shape",
    cpp_sources=["std::vector<int64_t> conv(torch::Tensor x, torch::Tensor w, torch::Tensor y, "
                 "int pad_h, int pad_w, int stride_h, int stride_w, "
                 "int dil_h, int dil_w, int groups, int algo, int math);"],
    cuda_sources=[SRC],
    functions=["conv"],
    verbose=False,
    with_cuda=True,
)


def bench(fn, warmup=5, trials=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / trials * 1000.0


SHAPES = {
    # name: (batch, in_ch, in_h, in_w, out_ch, k, stride, pad)
    "P50 (256x3x224x224 k11 s4 p2)": (256, 3, 224, 224, 96, 11, 4, 2),
    "P55 (8x64x512x1024 k3 s1 p1)":  (8, 64, 512, 1024, 128, 3, 1, 1),
    "P62 (8x64x512x256 k3 s1 p1)":   (8, 64, 512, 256, 128, 3, 1, 1),
    "small (8x64x64x64 k3 s1 p1)":   (8, 64, 64, 64, 128, 3, 1, 1),
}


def main():
    # THE GATE'S REFERENCE: eager fp32 with TF32 ON (C500 default).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = []
    for name, (b, c, h, w, oc, k, s, p) in SHAPES.items():
        torch.manual_seed(0)
        x = torch.randn(b, c, h, w, device="cuda")
        wt = torch.randn(oc, c, k, k, device="cuda")
        with torch.no_grad():
            ref = F.conv2d(x, wt, stride=s, padding=p)
        out_h = (h + 2 * p - k) // s + 1
        out_w = (w + 2 * p - k) // s + 1
        y = torch.zeros(b, oc, out_h, out_w, device="cuda")

        with torch.no_grad():
            eager_ms = bench(lambda: F.conv2d(x, wt, stride=s, padding=p))

        row = {"shape": name, "eager_ms": round(eager_ms, 4), "combos": []}
        print(f"\n{name}")
        print(f"  eager fp32 (TF32 ON): {eager_ms:.3f} ms  <-- the number to beat")
        for algo, aname in ((1, "PRECOMP_GEMM"), (5, "FFT_TILING")):
            for math, mname in ((0, "DEFAULT/TF32"), (1, "FMA/exact")):
                r = _mod.conv(x, wt, y, p, p, s, s, 1, 1, 1, algo, math)
                st, ws = r
                if st != 0:
                    print(f"    {aname:<12} {mname:<13}  status={st} (unavailable)")
                    row["combos"].append({"algo": aname, "math": mname,
                                          "status": st})
                    continue
                md = (y - ref).abs().max().item()
                ms = bench(lambda: _mod.conv(x, wt, y, p, p, s, s, 1, 1, 1,
                                             algo, math))
                ok = md < 1e-4
                print(f"    {'OK ' if ok else 'BAD'} {aname:<12} {mname:<13} "
                      f"maxdiff={md:.3e}  {ms:.3f} ms  ({eager_ms / ms:.2f}x)")
                row["combos"].append({"algo": aname, "math": mname,
                                      "status": 0, "maxdiff": md,
                                      "ms": round(ms, 4),
                                      "speedup": round(eager_ms / ms, 3)})
        results.append(row)

    out = Path(__file__).resolve().parent / "runs" / "conv_gate_shape_probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
