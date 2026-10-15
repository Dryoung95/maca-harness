# L1 conv family — no win, four measured negative results

The sibling `kda-on-maca` backlog named the one remaining performance
direction: "the convolutional networks need an mcdnn-based path, a different
kernel family entirely." This document is the result of opening that line.
It is a negative result, but a load-bearing one: the reason is not that the
right trick was not found, it is that the eager conv path and the only
gate-correct mcdnn path are the *same computation*.

Gate = `kernelbench.eval_kernel_against_ref`, the correctness authority.
Reference is eager fp32 **with TF32 on** (C500 ships `allow_tf32=True`). Note
the tension with the sibling repo's TF32 guidance — "any fp32 correctness
verdict that does not first disable TF32 may be reporting a precision policy
as a backend bug" — which is right for *diagnosing* a backend defect but is
the wrong yardstick for this gate: the reference itself is TF32-on, so a
candidate measured against a TF32-off reference is being compared to a
stricter standard than the one that grades it. Finding 2 lands exactly there.

## Setup

The board has the full mcdnn stack — `/opt/maca-3.3.0/lib/libmcdnn.so`
plus 6 fatbinaries, header `mcdnn.h` literally `#define CUDNN_MAJOR 8 /
MINOR 4`, an exact cuDNN 8.4.1 API port. Torch links `libmcdnn.so`
directly (`ldd` on `libbackend_with_compiler.so` et al.), which is the
first hint and later the whole story.

## Finding 1 — the SDK has no usable fp16 conv path

The GEMM line's win came from fp16 *operands* (mcblas fp16-in/fp32-out is
1.69x on square matmul), so the obvious transfer is an fp16 conv. It does
not exist (`probe_conv_f16.py`, descriptors matched to their buffers — an
earlier probe paired an fp32 buffer with a HALF descriptor and produced
pure bit-pattern garbage, 4e10, which is a probe bug, not a finding):

| operand types | result |
|---|---|
| f16 → f32 | `MCDNN_STATUS_BAD_PARAM` from the forward call itself |
| f16 → f16 | executes, maxdiff **5.7e-2** — wrong, not imprecise |

So the only correct operand type for conv on this SDK is fp32, and the
GEMM line's central trick does not transfer.

## Finding 2 — direct mcdnn cannot beat eager because it IS eager

With fp32 the only operand type, all 8 forward algorithms were probed
(via `probe_conv_bisect.py`, then on real L1 shapes). Exactly **two** are
instantiated — `IMPLICIT_PRECOMP_GEMM` and `FFT_TILING`; the other six
(`IMPLICIT_GEMM`, `GEMM`, `DIRECT`, `FFT`, `WINOGRAD`, `WINOGRAD_NONFUSED`)
all return `MCDNN_STATUS_NOT_SUPPORTED` on the workspace query. Of the two:

| algo | math | maxdiff vs eager (TF32 on) | time vs eager |
|---|---|---|---|
| PRECOMP_GEMM | DEFAULT (TF32 kernels) | **0.000e+00** | **1.00x** |
| PRECOMP_GEMM | FMA (exact fp32) | 4.5e-2 | 0.25x |
| FFT_TILING | DEFAULT (TF32 kernels) | 1.4e-1 | 0.69x |
| FFT_TILING | FMA (exact fp32) | 4.5e-2 | 0.53x |

 (`probe_conv_gate_shape.py`, shapes P50 256×3×224×224 k11s4p2, P55
 8×64×512×1024 k3, P62 8×64×512×256 k3, and a small control.)

The one configuration that passes the gate is a **tie at maxdiff exactly
zero**, and that is the whole point: `PRECOMP_GEMM + DEFAULT_MATH` selects
the TF32 kernels, eager conv makes the same selection through the same
`libmcdnn.so` torch already links, and the two are bit-identical because
they are the same computation on the same hardware path. Corroborated
directly: flipping `allow_tf32` changes eager conv's output by ~1.4e-3
(TF32-magnitude), so eager is provably on the TF32 path.

The FMA (exact) variants are *more* precise than the reference and therefore
**fail the 1e-4 gate** — the reference is TF32, so a candidate that removes
the TF32 approximation measures as an error. This is the same trap the
sibling repo's TF32 guidance documents, arrived at from the opposite
direction.

A first-attempt probe that compared against a TF32-*off* reference got
1.26e-2 "error" for the bit-identical-to-eager configuration — that number
was the reference's own TF32 policy, not a defect, and every number in this
document is measured against TF32-on because that is what the gate grades.

## Finding 3 — im2col + fp16 GEMM decomposition loses on both axes

Since direct mcdnn is a dead end, the remaining way to import the GEMM
line's win is to decompose the conv: `F.unfold` to build the im2col matrix,
then the fp16 GEMM that won L1 (`probe_conv_im2col.py`):

| shape | eager (TF32 on) | unfold | fp16 GEMM | sum | speedup | maxdiff |
|---|---:|---:|---:|---:|---:|---:|
| P50 (256×3×224×224 k11s4p2) | 6.9 ms | 15.9 ms | 1.8 ms | 17.7 ms | 0.39x | 3.3e-2 |
| P55 (8×64×512×1024 k3) | 9.9 ms | 20.4 ms | 4.4 ms | 24.8 ms | 0.40x | 6.2e-2 |
| P62 (8×64×512×256 k3) | 2.5 ms | 5.2 ms | 1.5 ms | 6.9 ms | 0.37x | 6.2e-2 |

It fails twice over:

* **Performance:** the unfold alone — a pure memory shuffle, no arithmetic —
  costs 2-3x the entire eager conv. The im2col matrix is (B, C·k·k, L): for
  P55 that is 8×576×524288 = 9.66 GB of fp32 traffic, and the write traffic
  dominates whatever the GEMM saves. The fp16 GEMM is fast (4.4 ms against
  the 9.9 ms conv), but it cannot see the shuffle around it.
* **Precision:** the contracted axis is C·k·k = 576 for these shapes, well
  under the ~8192 where the GEMM line located its precision boundary, so
  the pass looked plausible on paper. In fact maxdiff is 3-6e-2 — an order
  of magnitude worse than the fp16 GEMM's own ~1.5e-5 operand rounding,
  because the reference is TF32 and the candidate is not.

## Finding 4 — the vendor's own sample is the ground truth here

Worth recording as method: the first probe (f32, all 8 algos) returned
maxdiff 3.7 on every working combination, which read like another D1/D3-class
silent-corruption defect in the SDK. Building and running the vendor's own
`conv_fprop` sample verbatim took three minutes and returned `Check ok:
relative_diff 1.869853e-07, eps 1.000000e-05` — which moved the
investigation from "mcdnn conv is broken" to "my probe disagrees with the
vendor's configuration" and located the real bugs in my own probe (the
dtype pairing, and then the TF32 yardstick). Same lesson as the mcblas
leading-dimension convention: establish the vendor's known-good path first,
then bisect away from it.

## Conclusion

The backlog's "mcdnn-based path" for the conv family is closed as a line of
work on this board: direct mcdnn is the eager path itself, the fp16 operand
trick does not exist for conv, and the im2col decomposition loses to the
memory traffic before the GEMM can matter. The 58/82 GEMM-family result
stands as the performance line's final number. What would move conv is a
vendor-side change (a working fp16 conv path, or Winograd kernels that are
actually instantiated), not a harness-side one.

## Reproducing the probes

All five probes are self-contained (`torch` + `load_inline` only; no
cross-file imports). From the repository root:

```bash
source env.sh            # required: puts mcdnn headers on the include path
python3 docs/conv-family/probe_conv_algo.py         #Finding 2 (8-algo sweep)
python3 docs/conv-family/probe_conv_bisect.py       #Finding 2 (vendor-config bisect)
python3 docs/conv-family/probe_conv_f16.py          #Finding 1 (dtype pairing)
python3 docs/conv-family/probe_conv_gate_shape.py   #Finding 2 (real L1 shapes, TF32-on ref)
python3 docs/conv-family/probe_conv_im2col.py       #Finding 3
```

Each writes its raw JSON next to itself under `docs/conv-family/runs/`
(`Path(__file__).resolve().parent / "runs"`), and those committed JSON files
are the numbers quoted in the tables above. JIT build caches land in
`~/.cache/torch_extensions` outside the repository.

`gate_conv_family.py` — the in-gate `eval_kernel_against_ref` driver that
motivated this line, and `candidates/fast_conv.py` (the FastConv2d wrapper
whose mcdnn call the probes characterize) remain in the task workspace at
`/data/task-gemm-perf/`, outside git by design: the candidate never
submitted, so only its probes and their measurements are published here.
The probes do not import it; `fast_conv.py` is copied into this directory
solely so the docstrings that reference it stay readable.
