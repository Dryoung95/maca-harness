#!/usr/bin/env python3
"""Reconcile probe_determinism's "deterministic=True -> exactly 0" claim
against probe_p76_verdict/scope, which see the opposite.

probe_determinism.py compared TWO SEPARATELY CONSTRUCTED models (ma vs mb,
weights drawn consecutively from one RNG stream -> weights DIFFER). If the
bad batches' output does not depend on the weights at all, ma and mb would
agree EXACTLY on garbage -> "maxdiff 0.0" while still being wrong.

This probe separates the three effects with ONE model and ONE fixed input:

  P1  probe_determinism's exact protocol (ma vs mb, consecutive construction)
      under det=0 and det=1, with a per-batch breakdown of where the
      difference lives.
  P2  same-weights repeat forward (the true reproducibility question),
      det=0 and det=1.
  P3  weight-independence of the garbage: conv with weight forced to zero
      (correct output is identically 0). Anything nonzero on a batch is pure
      defect; whether the full-tensor path emits the SAME nonzero garbage
      for unrelated weights decides the "identical garbage" hypothesis.

Method validation: im2col vs conv on a small tensor, as before.

Run:  source env.sh && python3 probe_p76_reconcile.py
"""
import gc

import torch
import torch.nn.functional as F

C_IN, C_OUT, K, STRIDE, DIL = 64, 128, 3, 3, 4
PROBES = [0, 1, 15, 16, 31, 47, 48, 62, 63]


def ref_im2col(weight, xb):
    cols = F.unfold(xb.unsqueeze(3), kernel_size=(K, 1), dilation=(DIL, 1),
                    stride=(STRIDE, 1))
    w2 = weight.detach().reshape(C_OUT, -1)
    return torch.einsum("ok,bkl->bol", w2, cols)


def hdr(s):
    print(f"\n{s}\n{'-' * len(s)}", flush=True)


def main():
    dev = torch.device("cuda:0")

    # ---- method validation -------------------------------------------------
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    conv = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                           bias=False).to(dev).eval()
    torch.manual_seed(7)
    small = torch.rand(2, C_IN, 4096, device=dev)
    with torch.no_grad():
        e = (conv(small) - ref_im2col(conv.weight, small)).abs().max().item()
    print(f"method validation (small tensor): {e:.3e}", flush=True)
    del small
    gc.collect(); torch.cuda.empty_cache()

    torch.manual_seed(123)
    x = torch.rand(64, C_IN, 524280, device=dev)

    # ---- P1: probe_determinism's protocol ----------------------------------
    hdr("P1  probe_determinism protocol: ma vs mb, consecutive construction")
    for det in (False, True):
        torch.backends.cudnn.deterministic = det
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        ma = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                             bias=False).to(dev).eval()
        mb = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                             bias=False).to(dev).eval()
        weq = (ma.weight - mb.weight).abs().max().item()
        with torch.no_grad():
            a = ma(x); b = mb(x)
            d = (a - b).abs()
            overall = d.max().item()
            per = [(p, d[p].max().item()) for p in PROBES]
        bad = [p for p, v in per if v < 1e-6]
        big = [(p, v) for p, v in per if v >= 1e-6]
        print(f"  det={int(det)}: |wa-wb|max={weq:.3e}  ma-vs-mb overall "
              f"maxdiff={overall:.3e}", flush=True)
        print(f"    batches where ma==mb exactly (<1e-6): {bad}", flush=True)
        if big:
            print(f"    batches where they differ: "
                  f"{[(p, f'{v:.2e}') for p, v in big]}", flush=True)
        del a, b, d, ma, mb
        gc.collect(); torch.cuda.empty_cache()

    # ---- P2: same-weights repeat forward -----------------------------------
    hdr("P2  same weights, repeat forward (true reproducibility)")
    for det in (False, True):
        torch.backends.cudnn.deterministic = det
        with torch.no_grad():
            f1 = conv(x).clone()
            f2 = conv(x).clone()
            d = (f1 - f2).abs()
            overall = d.max().item()
            per = [(p, d[p].max().item()) for p in PROBES]
        print(f"  det={int(det)}: repeat-forward maxdiff = {overall:.3e}",
              flush=True)
        print(f"    per-probe-batch: "
              f"{[(p, f'{v:.2e}') for p, v in per]}", flush=True)
        del f1, f2, d
        gc.collect(); torch.cuda.empty_cache()

    # ---- P3: weight-independence of the garbage ----------------------------
    hdr("P3  weight-independence: zero-weight conv (correct output is 0)")
    zero = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                           bias=False).to(dev).eval()
    with torch.no_grad():
        zero.weight.zero_()
        full = zero(x)
        zmax = full.abs().max().item()
        zprobe = [(p, full[p].abs().max().item()) for p in PROBES]
        # per-batch control: the zero-weight conv must be exactly 0
        ctrl = max((zero(x[b:b + 1]).abs().max().item()) for b in (0, 63))
    print(f"  zero-weight full-tensor conv: |out|max = {zmax:.3e}", flush=True)
    print(f"  per-probe-batch |out|max: "
          f"{[(p, f'{v:.2e}') for p, v in zprobe]}", flush=True)
    print(f"  zero-weight per-batch control: |out|max = {ctrl:.3e}",
          flush=True)
    del full, zero
    gc.collect(); torch.cuda.empty_cache()

    hdr("P4  summary of what each face establishes")
    print("  P1 separates weight effects from forward effects.", flush=True)
    print("  P2 is the reproducibility question the documentation needs.",
          flush=True)
    print("  P3 tells us whether bad-batch output is weight-independent "
          "garbage.", flush=True)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
