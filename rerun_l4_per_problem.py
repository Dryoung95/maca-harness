#!/usr/bin/env python3
"""Re-measure Level 4 with one fresh process per problem.

Two attempts at measuring L4 in the same process as other work bothFailed
the same way: after enough problems the PyTorch caching allocator holds
~60 GB of live blocks, and L4's multi-GB models OOM on the allocator
rather than the card. First inside the 270-problem sweep (250 problems of
L1-L3 before L4), then again in a clean process running only L4 — L4's
own models are large enough that the allocator fills up *within* the level
(P1 fine, P3/P4/P8/P10 OOM by the tenth problem).

L4 is also the level where isolation matters most for the number's sake:
a baseline timed under allocator pressure is not comparable to anything a
candidate kernel will be measured against, and KernelBench's own timing
loop is inside one process.

So: one subprocess per problem, 20 fresh interpreters. Each inherits the
build cache, so nothing is recompiled. A problem that still OOMs in a
pristine process is a genuine capacity limit and is reported as such.

    python3 rerun_l4_per_problem.py [--out perf-baseline-l4-clean.jsonl]

Writes the 20 rows, then merges into perf-baseline.jsonl (L1-L3 from the
sweep, L4 from here) and calls finalize_perf_baseline.py.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/data/maca-harness")
MAIN = REPO / "perf-baseline.jsonl"
ENV = REPO / "env.sh"
RUN_BATCH = REPO / "run_batch.py"
BUILD_ROOT = REPO / "perf_build"


def measure_one(pid: int, out_path: Path) -> dict | None:
    """Run one L4 problem in a fresh interpreter; return its record."""
    # Online model-update checks make transformers block in a kernel
    # select() on port 443 for minutes at a time, with no timeout of its
    # own -- that hung P14 for 20 minutes mid-sweep. The weights are
    # already in the local HF cache, so force offline and cap each problem
    # well above its observed 100 s cost.
    cmd = (
        f"source {ENV} && "
        f"export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 && "
        f"exec {sys.executable} {RUN_BATCH} --levels 4 --problem-ids {pid} "
        f"--num-correct-trials 5 --num-perf-trials 20 "
        f"--out {out_path} --build-root {BUILD_ROOT}"
    )
    t0 = time.time()
    try:
        r = subprocess.run(["/bin/bash", "-c", cmd], cwd=str(REPO),
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"  P{pid:>2} TIMEOUT after 900s (killed; likely a hung "
              f"network select)", flush=True)
        return None
    dt = time.time() - t0

    # run_batch appends one line per problem; take the last line it wrote
    # for this pid.
    recs = [json.loads(l) for l in open(out_path) if l.strip()]
    mine = [x for x in recs if x.get("level") == 4 and x.get("problem_id") == pid]
    if not mine:
        print(f"  P{pid:>2} NO RECORD after {dt:.0f}s  "
              f"(exit {r.returncode}: {r.stderr[-200:]})", flush=True)
        return None
    rec = mine[-1]
    sp = rec.get("speedup")
    print(f"  P{pid:>2} {rec['status']:<8} spd={sp if sp is not None else 'n/a':<8} "
          f"{dt:.0f}s  {rec.get('name', '')[:44]}", flush=True)
    if rec["status"] != "passed":
        print(f"       {rec.get('failure_class')}: {str(rec.get('error'))[:110]}",
              flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "perf-baseline-l4-clean.jsonl"))
    ap.add_argument("--force", action="store_true",
                    help="discard the existing partial run and redo all 20")
    args = ap.parse_args()
    out_path = Path(args.out)

    # Resume support: a hung subprocess can stop the sweep partway, and the
    # finished rows are expensive to re-derive (each is a full L4 eval).
    # Skip problems whose record is already in the file unless --force.
    done = set()
    if out_path.exists() and not args.force:
        for line in open(out_path):
            if line.strip():
                rec = json.loads(line)
                if rec.get("level") == 4:
                    done.add(rec["problem_id"])
    elif out_path.exists():
        out_path.unlink()

    todo = [p for p in range(1, 21) if p not in done]
    print(f"measuring L4 one process per problem "
          f"({len(todo)} to do, {len(done)} already collected)\n", flush=True)
    recs = []
    for pid in todo:
        rec = measure_one(pid, out_path)
        if rec:
            recs.append(rec)

    # The merge reads every L4 row from the output file, not just the ones
    # collected in this invocation: in resume mode the earlier rows were
    # written by a previous run of this same script.
    all_l4 = [json.loads(l) for l in open(out_path) if l.strip()]
    all_l4 = [r for r in all_l4 if r.get("level") == 4]
    ok = [r for r in all_l4 if r["status"] == "passed"]
    print("\n" + "=" * 70)
    print(f"{len(ok)}/{len(all_l4)} L4 problems passed in isolated processes")

    if len(all_l4) < 20:
        print("WARNING: some problems produced no record; see above")

    # Merge: L1-L3 from the sweep, L4 from this isolated run.
    main_recs = [json.loads(l) for l in open(MAIN) if l.strip()]
    keep = [r for r in main_recs if r["level"] != 4]
    print(f"[merge] {len(main_recs)} sweep rows -> {len(keep)} L1-L3 kept, "
          f"{len(main_recs) - len(keep)} L4 replaced by isolated rows")

    # Preserve L4 ordering by problem_id regardless of collection order.
    all_l4.sort(key=lambda r: r["problem_id"])
    merged = keep + all_l4
    backup = MAIN.with_suffix(".jsonl.premerge")
    shutil.copy(MAIN, backup)
    with open(MAIN, "w") as f:
        for r in merged:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"[merge] wrote {len(merged)} rows to {MAIN} (backup: {backup.name})")

    subprocess.run([sys.executable, str(REPO / "finalize_perf_baseline.py")])


if __name__ == "__main__":
    main()
