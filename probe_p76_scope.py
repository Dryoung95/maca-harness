#!/usr/bin/env python3
"""Reconcile the apparent contradiction between the two P76 probes.

probe_determinism.py reported: with cudnn.deterministic=True, a
reference-vs-reference comparison (two model instances, ma vs mb) gives
maxdiff EXACTLY 0.0 for L1 P76.
probe_p76_verdict.py reported: under the SAME deterministic flag, two
back-to-back full-tensor forwards disagree by maxdiff 1.0.

Both are true if the flags affect two different things:
  * deterministic=True may fix RNG-consuming weight initialization so ma and
    mb get identical weights AND identical (equally wrong) conv outputs ->
    ma-vs-mb is 0 without the forward being correct or even reproducible.
  * the conv forward itself stays non-reproducible at this scale.

This script measures all four faces of the same setup in ONE process so the
comparison is apples-to-apples, and additionally asks the question neither
probe could: is the per-batch path merely *less wrong*, or exact?

Faces measured on the real P76 shape (64, 64, 524280), 5 reps where useful:
  1. ma-vs-mb, det=0 (probe_determinism's protocol)
  2. ma-vs-mb, det=1 (the reported "exactly 0" case)
  3. same-model repeat forward, det=0
  4. same-model repeat forward, det=1
  5. full-tensor vs sum-of-per-batch, det=0 and det=1, vs im2col truth

Run:  source env.sh && python3 probe_p76_scope.py
"""
import gc

import torch
import torch.nn.functional as F

C_IN, C_OUT, K, STRIDE, DIL = 64, 128, 3, 3, 4
PROBES = [0, 1, 16, 31, 47, 48, 62, 63]
REPS = 3


def ref_im2col(conv, xb):
    cols = F.unfold(xb.unsqueeze(3), kernel_size=(K, 1), dilation=(DIL, 1),
                    stride=(STRIDE, 1))
    w2 = conv.weight.detach().reshape(C_OUT, -1)
    return torch.einsum("ok,bkl->bol", w2, cols)


def hdr(s):
    print(f"\n{s}\n{'-' * len(s)}")


def main():
    dev = torch.device("cuda:0")

    hdr("1. weight equality: does deterministic fix weight init? (probe_determinism protocol)")
    for det in (False, True):
        torch.backends.cudnn.deterministic = det
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        ma = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                             bias=False).to(dev).eval()
        mb = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                             bias=False).to(dev).eval()
        weq = (ma.weight - mb.weight).abs().max().item()
        print(f"  det={int(det)}: |wa - wb|max = {weq:.3e}"
              f"  -> weights {'IDENTICAL' if weq == 0 else 'DIFFER'}")
        del ma, mb
        gc.collect(); torch.cuda.empty_cache()

    # weight init does not consume the CUDA RNG in a way the cudnn flag could
    # plausibly gate; whatever we measure above is what the earlier probe saw.

    hdr(f"2. same-model repeat forward, {REPS} reps (stability of the forward itself)")
    torch.manual_seed(42)
    conv = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                           bias=False).to(dev).eval()
    torch.manual_seed(123)
    x = torch.rand(64, C_IN, 524280, device=dev)

    for det in (False, True):
        torch.backends.cudnn.deterministic = det
        with torch.no_grad():
            outs = [conv(x)[PROBES].clone() for _ in range(REPS)]
            for i in range(1, REPS):
                d = (outs[0] - outs[i]).abs().max().item()
                print(f"  det={int(det)}: forward0 vs forward{i}: maxdiff = {d:.3e}")
        del outs
        gc.collect(); torch.cuda.empty_cache()

    hdr("3. full-tensor vs per-batch accumulation (same weights, same input)")
    for det in (False, True):
        torch.backends.cudnn.deterministic = det
        with torch.no_grad():
            full = conv(x)[PROBES].clone()
            per = torch.stack([conv(x[b:b + 1]) for b in PROBES]).squeeze(1)
            d = (full - per).abs().max().item()
            w_wrong = (full - per).abs()
            n_bad = int((w_wrong > 1e-4).sum().item())
            print(f"  det={int(det)}: full vs per-batch: maxdiff = {d:.3e}, "
                  f"mismatched elements {n_bad:,}")
        del full, per, w_wrong
        gc.collect(); torch.cuda.empty_cache()

    hdr("4. which of the two faces the im2col adjudicator agrees with")
    for det in (False, True):
        torch.backends.cudnn.deterministic = det
        with torch.no_grad():
            full = conv(x)[PROBES].clone()
            per = torch.stack([conv(x[b:b + 1]) for b in PROBES]).squeeze(1)
            err_f = max((full[i] - ref_im2col(conv, x[b:b + 1])).abs().max().item()
                        for i, b in enumerate(PROBES))
            err_p = max((per[i] - ref_im2col(conv, x[b:b + 1])).abs().max().item()
                        for i, b in enumerate(PROBES))
            print(f"  det={int(det)}: full vs im2col = {err_f:.3e} "
                  f"({'ok' if err_f < 1e-4 else 'WRONG'})   "
                  f"per-batch vs im2col = {err_p:.3e} "
                  f"({'ok' if err_p < 1e-4 else 'WRONG'})")
        del full, per
        gc.collect(); torch.cuda.empty_cache()

    hdr("5. scale threshold: at what batch/element count does full-tensor break?")
    torch.backends.cudnn.deterministic = False
    with torch.no_grad():
        for bs in (1, 2, 4, 8, 16, 32, 64):
            xb = x[:bs]
            full = conv(xb)
            per = torch.cat([conv(xb[b:b + 1]) for b in range(bs)], 0)
            d = (full - per).abs().max().item()
            n = xb.numel()
            print(f"  batch {bs:>2} ({n:>13,} elems): full vs per maxdiff = "
                  f"{d:.3e} {'<-- WRONG' if d > 1e-4 else ''}")
            del full, per, xb
            gc.collect(); torch.cuda.empty_cache()

    torch.backends.cudnn.deterministic = False
    print("\nreconciled reading appears in section 4/5 above.")


if __name__ == "__main__":
    main()
