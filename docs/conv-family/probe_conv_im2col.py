#!/usr/bin/env python3
"""Time the im2col + fp16 GEMM conv decomposition against eager.

probe_conv_gate_shape.py closed the direct-mcdnn angle: eager conv already
runs through the same libmcdnn.so, the only gate-correct configuration
(PRECOMP_GEMM + TF32) IS the eager computation (maxdiff 0.000, 1.00x), and
the SDK has no usable fp16 conv path. So there is no win from calling mcdnn
ourselves.

One angle remains: decompose the conv as im2col + GEMM and run the GEMM
through the fp16 path that won the GEMM line (mcblas fp16-in/fp32-out,
1.69x on square matmul). The im2col (F.unfold) is a pure memory shuffle —
it reads the input once and writes a (B, C*k*k, L) matrix — so the question
is whether the GEMM saving survives the unfold traffic.

Precision: the contracted axis here is C*k*k (576 for a 64-channel 3x3
conv), well below the ~8192 where the GEMM line found the fp16 precision
boundary, so this has a real chance of passing the 1e-4 gate.

Reference is eager fp32 with TF32 on — what the gate actually grades
against. Anything compared to a TF32-off reference measures the wrong
yardstick (see probe_conv_gate_shape.py).
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def bench(fn, warmup=5, trials=20):
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
}


def main():
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

        print(f"\n{name}")
        with torch.no_grad():
            eager_ms = bench(lambda: F.conv2d(x, wt, stride=s, padding=p))
        print(f"  eager fp32 (TF32 ON): {eager_ms:.3f} ms  <-- the number to beat")

        # im2col: (B, C*k*k, out_h*out_w)
        w2c = c * k * k
        with torch.no_grad():
            def im2col_gemm_fp16():
                col = F.unfold(x, k, padding=p, stride=s)  # (B, Ckk, L)
                bl, l = col.size(0), col.size(2)
                col2 = col.permute(0, 2, 1).reshape(bl * l, w2c)
                # fp16 operands, fp32 accumulate via fp32 output
                out = (col2.half() @ wt.reshape(oc, w2c).half().t()).float()
                return out.view(b, out_h, out_w, oc).permute(0, 3, 1, 2)

            # correctness first
            y = im2col_gemm_fp16()
            md = (y - ref).abs().max().item()
            ok = md < 1e-4
            print(f"    {'OK ' if ok else 'BAD'} im2col+fp16-GEMM  maxdiff={md:.3e}")
            if ok:
                ms = bench(im2col_gemm_fp16)
                print(f"         {ms:.3f} ms  ({eager_ms / ms:.2f}x)")
                results.append({"shape": name, "maxdiff": md, "ms": round(ms, 4),
                                "speedup": round(eager_ms / ms, 3)})
            else:
                results.append({"shape": name, "maxdiff": md})

            # component breakdown to explain the number
            col = F.unfold(x, k, padding=p, stride=s)
            unfold_ms = bench(lambda: F.unfold(x, k, padding=p, stride=s))
            col2 = col.permute(0, 2, 1).reshape(b * col.size(2), w2c)
            wh = wt.reshape(oc, w2c).half().t().contiguous()
            colh = col2.half().contiguous()
            gemm_ms = bench(lambda: colh @ wh)
            print(f"    components: unfold {unfold_ms:.3f} ms + "
                  f"fp16 GEMM {gemm_ms:.3f} ms = "
                  f"{unfold_ms + gemm_ms:.3f} ms vs eager {eager_ms:.3f} ms")
            print(f"    im2col matrix: {tuple(col.shape)} = "
                  f"{col.numel() * 4 / 1e9:.2f} GB fp32")

    out = (Path(__file__).resolve().parent / "runs"
           / "conv_im2col_probe.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
