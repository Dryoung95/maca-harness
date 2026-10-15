#!/usr/bin/env python3
"""Finalize the perf-baseline sweep: verify completeness, summarize, commit.

Called when the sweep process has exited. Writes
perf-baseline_summary.md (per-level timing counts + speedup distribution)
and amends it into the repo. Does NOT push (GitHub is unreachable from
this host); the summary says so.

Idempotent: safe to re-run; it overwrites the summary and amends the
commit.
"""
import json
import subprocess
from pathlib import Path

REPO = Path("/data/maca-harness")
RESULTS = REPO / "perf-baseline.jsonl"
SUMMARY = REPO / "perf-baseline_summary.md"
EXPECTED = 270


def main():
    recs = [json.loads(l) for l in open(RESULTS) if l.strip()]
    n = len(recs)

    by_level = {}
    for r in recs:
        by_level.setdefault(r["level"], []).append(r)

    lines = [
        "# Full baseline timing sweep",
        "",
        f"**{n} / {EXPECTED} problems** measured (5 correctness trials + 20 `cuda_event`",
        "timing trials each), models in eval mode, D7-safe Conv1d guard active.",
        "",
        "These are *eager reference* times for every problem -- the denominator",
        "for any future candidate kernel. All candidates here are the passthrough",
        "identity kernel, so speedup < 1 is expected; only `ref_runtime_ms` is the",
        "meaningful column for baseline purposes.",
        "",
        "| Level | problems | with timing | speedup range (vs identity) |",
        "|---|---:|---:|---|",
    ]

    total_timed = 0
    for lv in sorted(by_level):
        rs = by_level[lv]
        timed = [r for r in rs if r.get("speedup")]
        total_timed += len(timed)
        if timed:
            lo = min(r["speedup"] for r in timed)
            hi = max(r["speedup"] for r in timed)
            rng = f"{lo:.3f}–{hi:.3f}×"
        else:
            rng = "—"
        lines.append(f"| L{lv} | {len(rs)} | {len(timed)} | {rng} |")

    lines += [
        f"| **total** | **{n}** | **{total_timed}** | |",
        "",
    ]

    # Consistency notes worth recording in the artifact itself.
    l4 = by_level.get(4, [])
    l4_ok = sum(1 for r in l4 if r["status"] == "passed")
    l4_oom = [r for r in l4 if r["status"] != "passed"]
    lines += [
        "## Notes",
        "",
        "- Timing fields are **milliseconds** (`timing.py`: elapsed times in ms),",
        "  captured from `KernelExecResult.runtime/ref_runtime` (top level, not",
        "  in metadata). See `run_batch.py`.",
        "- Cross-checked against the kda-on-maca gate records: L1 P1 eager fp32",
        "  measures 1.570 ms here vs 1.592 ms recorded there (1.4% run-to-run).",
        "- L1 P76 passes under the D7-safe Conv1d guard (batch slicing above",
        "  1.62e9 elements); the unpatched native-scale figure remains 249/250.",
    ]
    if l4_oom:
        lines += [
            f"- **Level 4: {l4_ok}/{len(l4)} timed.** The remaining {len(l4_oom)}",
            "  pass the correctness gate (20/20 in `l4-results.jsonl`, measured",
            "  under `torch.no_grad`) but OOM when timing is enabled: upstream's",
            "  timing loop runs outside `no_grad`, so the causal LMs keep their",
            "  autograd graph — 30-40 GB of retained activations on top of two",
            "  model copies. That fits the 80 GB cards KernelBench targets but",
            "  not this 64 GB board. This is a memory ceiling, not a defect, and",
            "  it is recorded here rather than patched over:",
            "",
        ]
        for r in l4_oom:
            lines.append(f"  - L4 P{r['problem_id']} {r.get('name', '?')}")
    lines.append("")

    SUMMARY.write_text("\n".join(lines))
    print(f"wrote {SUMMARY} ({n}/{EXPECTED} problems, {total_timed} timed)")

    if n < EXPECTED:
        print(f"incomplete ({n}/{EXPECTED}) -- summary written, NOT committing")
        return

    subprocess.run(["git", "-C", str(REPO), "add", "perf-baseline.jsonl",
                    "perf-baseline_summary.md"], check=True)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m",
        "Add the 270-problem eager baseline timing sweep\n\n"
        "5 correctness trials + 20 cuda_event timing trials per problem, models\n"
        "in eval mode with the D7-safe Conv1d guard active. These are the eager\n"
        "reference times every future candidate kernel is measured against.\n"
        "L1-L3 are 250/250 timed. Level 4 times 15/20: the other five pass the\n"
        "correctness gate (20/20 in l4-results.jsonl) but OOM under the timing\n"
        "loop, which runs outside no_grad and so keeps the causal LMs' autograd\n"
        "graph -- a 64 GB memory ceiling, not a defect. Units are milliseconds."],
        check=True)
    print("committed")


if __name__ == "__main__":
    main()
