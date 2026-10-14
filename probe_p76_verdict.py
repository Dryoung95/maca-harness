#!/usr/bin/env python3
"""Decisive adjudication of L1 P76: is deterministic-mode conv CORRECT or
just CONSISTENTLY WRONG?

Prior findings (task-gemm-perf/probe_p76_adjudicator.py):
  * im2col + GEMM adjudicator agrees with conv on small tensors (4.8e-7)
  * per-batch conv (one batch slice, <= 33.5M elements) agrees with im2col
    on EVERY probe batch (~7e-7)
  * full-tensor conv disagrees with im2col by up to 1.95 on the bad batches
  * ref-vs-ref divergence of the full-tensor path collapses to 0 under
    torch.backends.cudnn.deterministic=True

That last fact is ambiguous: two identical garbage results also differ by
0. This script settles it by comparing each path against the im2col
adjudicator, which never runs the conv kernel:

  A. full-tensor conv, default (nondeterministic) algo selection
  B. full-tensor conv, cudnn.deterministic=True   <- the open question
  C. full-tensor conv, cudnn.benchmark=True
  D. per-batch conv (one batch at a time)         <- known-correct scale

If B == im2col, deterministic mode selects a correct algorithm and P76 has
a clean workaround. If B != im2col but B is run-to-run stable, the
deterministic path is consistently wrong and P76 is a genuine platform
defect at this tensor scale -- correctly reported as a failure.

Run:  source env.sh && python3 probe_p76_verdict.py
"""
import gc

import torch
import torch.nn.functional as F

C_IN, C_OUT, K, STRIDE, DIL = 64, 128, 3, 3, 4
PROBES = [0, 1, 16, 31, 47, 48, 62, 63]


def ref_im2col(conv, xb):
    """Adjudicator: unfold + GEMM, never invokes the conv kernel."""
    cols = F.unfold(xb.unsqueeze(3), kernel_size=(K, 1), dilation=(DIL, 1),
                    stride=(STRIDE, 1))
    w2 = conv.weight.detach().reshape(C_OUT, -1)
    return torch.einsum("ok,bkl->bol", w2, cols)


def run_full(conv, x, det, bench):
    """Full-tensor forward under a chosen backend mode; returns probe slices."""
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = bench
    with torch.no_grad():
        out = conv(x)
        keep = out[PROBES].clone()
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return keep


def score(conv, x, keep, label):
    """Compare stored probe slices against the per-batch im2col truth."""
    worst, n_bad = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(PROBES):
            truth = ref_im2col(conv, x[b:b + 1])
            err = (keep[i] - truth).abs().max().item()
            worst = max(worst, err)
            n_bad += err >= 1e-4
            del truth
            gc.collect()
            torch.cuda.empty_cache()
    tag = "CORRECT" if worst < 1e-4 else "WRONG"
    print(f"{label:<44} vs im2col: maxdiff {worst:.3e}  bad batches {n_bad}/{len(PROBES)}  {tag}")
    return worst, n_bad


def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(42)
    conv = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                           bias=False).to(dev).eval()

    # validate the adjudicator at a scale where nothing is in question
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    small = torch.rand(2, C_IN, 4096, device=dev)
    with torch.no_grad():
        e = (conv(small) - ref_im2col(conv, small)).abs().max().item()
    print(f"method validation (small tensor): {e:.3e}")
    assert e < 1e-4, "adjudicator disagrees with conv; cannot judge"
    del small
    gc.collect(); torch.cuda.empty_cache()

    torch.manual_seed(123)
    x = torch.rand(64, C_IN, 524280, device=dev)
    print(f"input (64, {C_IN}, 524280) = {x.numel():,} elements "
          f"(int32 ceiling {2**31:,}, headroom {2**31 - x.numel():,})\n")

    results = {}

    # A. default algo selection
    keep_a = run_full(conv, x, det=False, bench=False)
    results["A default"] = score(conv, x, keep_a, "A full-tensor, default algos")
    del keep_a; gc.collect(); torch.cuda.empty_cache()

    # B. deterministic — the open question. Run twice: does it also reproduce
    #    run-to-run (stability) and does it match truth (correctness)?
    keep_b1 = run_full(conv, x, det=True, bench=False)
    results["B deterministic"] = score(conv, x, keep_b1, "B full-tensor, cudnn.deterministic=True")
    keep_b2 = run_full(conv, x, det=True, bench=False)
    with torch.no_grad():
        repro = (keep_b1 - keep_b2).abs().max().item()
    print(f"{'B run-to-run reproducibility':<44} maxdiff {repro:.3e}  "
          f"{'STABLE' if repro < 1e-6 else 'UNSTABLE'}")
    results["B repro"] = (repro, 0)
    del keep_b1, keep_b2; gc.collect(); torch.cuda.empty_cache()

    # C. benchmark mode (caches algo selection per shape)
    keep_c = run_full(conv, x, det=False, bench=True)
    results["C benchmark"] = score(conv, x, keep_c, "C full-tensor, cudnn.benchmark=True")
    del keep_c; gc.collect(); torch.cuda.empty_cache()

    # D. per-batch — known-correct scale, as a control
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        per = torch.stack([conv(x[b:b + 1]) for b in PROBES]).squeeze(1)
    results["D per-batch"] = score(conv, x, per, "D per-batch conv (control)")
    del per; gc.collect(); torch.cuda.empty_cache()

    print("\nverdict:")
    b_worst = results["B deterministic"][0]
    d_worst = results["D per-batch"][0]
    if b_worst < 1e-4:
        print("  deterministic mode selects a CORRECT algorithm at full scale")
        print("  -> P76 can pass the official gate via a harness-layer")
        print("     torch.backends.cudnn.deterministic=True, and the pass is real")
    elif d_worst < 1e-4 and b_worst >= 1e-4:
        print("  deterministic mode is stable but still WRONG at full scale")
        print("  -> P76 is a genuine MACA large-tensor conv defect;")
        print("     correctly reported as the sole real failure")
    else:
        print("  unexpected: per-batch control itself disagrees with im2col")
        print("  -> adjudicator invalid at this scale; needs narrower probes")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
