# Full baseline timing sweep

**270 / 270 problems** measured (5 correctness trials + 20 `cuda_event`
timing trials each), models in eval mode, D7-safe Conv1d guard active.

These are *eager reference* times for every problem -- the denominator
for any future candidate kernel. All candidates here are the passthrough
identity kernel, so speedup < 1 is expected; only `ref_runtime_ms` is the
meaningful column for baseline purposes.

| Level | problems | with timing | speedup range (vs identity) |
|---|---:|---:|---|
| L1 | 100 | 100 | 0.171–1.000× |
| L2 | 100 | 100 | 0.475–1.000× |
| L3 | 50 | 50 | 0.614–1.126× |
| L4 | 20 | 12 | 0.801–1.011× |
| **total** | **270** | **262** | |

## Notes

- Timing fields are **milliseconds** (`timing.py`: elapsed times in ms),
  captured from `KernelExecResult.runtime/ref_runtime` (top level, not
  in metadata). See `run_batch.py`.
- Cross-checked against the kda-on-maca gate records: L1 P1 eager fp32
  measures 1.570 ms here vs 1.592 ms recorded there (1.4% run-to-run).
- L1 P76 passes under the D7-safe Conv1d guard (batch slicing above
  1.62e9 elements); the unpatched native-scale figure remains 249/250.
- **Level 4: 12/20 timed.** The remaining 8
  pass the correctness gate (20/20 in `l4-results.jsonl`, measured
  under `torch.no_grad`) but OOM when timing is enabled: upstream's
  timing loop runs outside `no_grad`, so the causal LMs keep their
  autograd graph — 30-40 GB of retained activations on top of two
  model copies. That fits the 80 GB cards KernelBench targets but
  not this 64 GB board. This is a memory ceiling, not a defect, and
  it is recorded here rather than patched over:

  - L4 P1 1_EleutherAI-gpt-neo-2p7B_bs32_seq256.py
  - L4 P3 3_EleutherAI-gpt-neo-2p7B_bs1_seq2047.py
  - L4 P4 4_facebook-opt-1p3b_bs32_seq256.py
  - L4 P8 8_facebook-opt-1p3b_bs512_seq32.py
  - L4 P10 10_google-bigbird-roberta-base_bs1024_seq32.py
  - L4 P17 17_facebook-bart-large_bs1024_seq32.py
  - L4 P18 18_EleutherAI-gpt-neo-2p7B_bs512_seq32.py
  - L4 P19 19_gpt2_bs1024_seq32.py
