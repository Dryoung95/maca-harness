#!/usr/bin/env python3
"""Probe the fp16 conv path with descriptors matched to their buffers.

probe_conv_algo.py's f16 rows were garbage (maxdiff ~4e10) for a boring
reason: the y descriptor said HALF while the buffer was fp32 — a type
mismatch that makes mcdnn write fp16 bits into a float array. The bisect
probe then showed the f32 path is exact only under math=FMA (MMA_F32
kernels); DEFAULT_MATH picks the broken TF32 kernels (2.4e-2 error).

The GEMM line's win came from fp16 *operands* (mcblas fp16-in/fp32-out is
1.69x), so the question this probe answers is whether mcdnn conv has an
equivalent fp16-in path that is both correct and fast. Descriptor and
buffer dtypes are paired here by construction.

Combinations tested (x/w dtype -> y dtype):
  f16 -> f32    the GEMM-line analogue (small operands, fp32 output)
  f16 -> f16    pure fp16, in case the epilogue rounding is tolerable
  f32 -> f32    exact baseline for the timing comparison

Two algos (the only two the earlier probe found instantiated), two math
modes, correctness against an fp32 eager reference with TF32 *off* (the
reference must be exact, or the comparison measures the wrong thing).
"""
import json
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

// xcode/wcode/ocode: 0=float, 1=half. algo and math as before.
// Returns [query_status, exec_status, workspace_bytes].
std::vector<int64_t> conv(torch::Tensor x, torch::Tensor w, torch::Tensor y,
                          int pad_h, int pad_w, int stride_h, int stride_w,
                          int dil_h, int dil_w, int groups,
                          int xcode, int wcode, int ocode,
                          int algo, int math) {
    std::vector<int64_t> out(3, -1);

    mcdnnDataType_t xdt = xcode == 0 ? MCDNN_DATA_FLOAT : MCDNN_DATA_HALF;
    mcdnnDataType_t wdt = wcode == 0 ? MCDNN_DATA_FLOAT : MCDNN_DATA_HALF;
    mcdnnDataType_t odt = ocode == 0 ? MCDNN_DATA_FLOAT : MCDNN_DATA_HALF;

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

    mcdnnSetTensor4dDescriptor(x_desc, MCDNN_TENSOR_NCHW, xdt, batch, in_ch, in_h, in_w);
    mcdnnSetFilter4dDescriptor(w_desc, wdt, MCDNN_TENSOR_NCHW, out_ch, in_ch / groups, k_h, k_w);
    mcdnnSetTensor4dDescriptor(y_desc, MCDNN_TENSOR_NCHW, odt, batch, out_ch, out_h, out_w);
    // computeType per mcdnn convention: the descriptor's data type
    mcdnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                    dil_h, dil_w, MCDNN_CROSS_CORRELATION, odt);
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

    mcdnnDestroyTensorDescriptor(x_desc);
    mcdnnDestroyTensorDescriptor(y_desc);
    mcdnnDestroyFilterDescriptor(w_desc);
    mcdnnDestroyConvolutionDescriptor(conv_desc);
    return out;
}
"""

_mod = load_inline(
    name="probe_conv_f16",
    cpp_sources=["std::vector<int64_t> conv(torch::Tensor x, torch::Tensor w, torch::Tensor y, "
                 "int pad_h, int pad_w, int stride_h, int stride_w, "
                 "int dil_h, int dil_w, int groups, "
                 "int xcode, int wcode, int ocode, int algo, int math);"],
    cuda_sources=[SRC],
    functions=["conv"],
    verbose=False,
    with_cuda=True,
)

DT = {0: torch.float32, 1: torch.float16}
ALGOS = {1: "PRECOMP_GEMM", 5: "FFT_TILING"}


def bench(fn, warmup=5, trials=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / trials * 1000.0  # ms


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    results = []
    # A shape big enough for timing to be meaningful but not to disturb
    # anything else that might be using the card.
    batch, in_ch, hw, out_ch, k = 8, 64, 64, 128, 3
    for label, (xc, wc, oc) in {
        "f32->f32": (0, 0, 0),
        "f16->f32": (1, 1, 0),
        "f16->f16": (1, 1, 1),
    }.items():
        torch.manual_seed(0)
        x32 = torch.randn(batch, in_ch, hw, hw, device="cuda")
        w32 = torch.randn(out_ch, in_ch, k, k, device="cuda")
        with torch.no_grad():
            ref = F.conv2d(x32, w32, padding=k // 2)
        x = x32.to(DT[xc])
        w = w32.to(DT[wc])

        for algo, aname in ALGOS.items():
            for math, mname in ((0, "DEFAULT"), (1, "FMA")):
                y = torch.zeros(ref.shape, device="cuda", dtype=DT[oc])
                r = _mod.conv(x, w, y, k // 2, k // 2, 1, 1, 1, 1, 1,
                              xc, wc, oc, algo, math)
                q, e, ws = r
                if q != 0 or e != 0:
                    print(f"  {label:<10} {aname:<12} {mname:<8}  "
                          f"q={q} e={e}  (unavailable)")
                    results.append({"cfg": f"{label}/{aname}/{mname}",
                                    "ok": False})
                    continue
                yf = y.float()
                md = (yf - ref).abs().max().item()
                ms = bench(lambda: _mod.conv(x, w, y, k // 2, k // 2, 1, 1,
                                             1, 1, 1, xc, wc, oc, algo, math)
                           )
                ok = md < 1e-3
                print(f"  {'OK ' if ok else 'BAD'} {label:<10} {aname:<12} "
                      f"{mname:<8} maxdiff={md:.3e}  {ms:.3f} ms")
                results.append({"cfg": f"{label}/{aname}/{mname}", "ok": ok,
                                "maxdiff": md, "ms": ms, "ws": ws})

    # eager comparison, TF32 off, same shape
    torch.manual_seed(0)
    x32 = torch.randn(batch, in_ch, hw, hw, device="cuda")
    w32 = torch.randn(out_ch, in_ch, k, k, device="cuda")
    with torch.no_grad():
        eager_ms = bench(lambda: F.conv2d(x32, w32, padding=k // 2))
    print(f"\n  eager fp32 (TF32 off): {eager_ms:.3f} ms  <-- the number to beat")
    for r in results:
        if r.get("ok") and r.get("ms"):
            print(f"    {r['cfg']:<34} {r['ms'] / eager_ms:.3f}x vs eager")

    out = Path(__file__).resolve().parent / "runs" / "conv_f16_probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"results": results, "eager_ms": eager_ms},
                              indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
