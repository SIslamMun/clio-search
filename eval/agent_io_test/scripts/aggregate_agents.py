#!/usr/bin/env python3
"""Aggregate per-rank agent run results into a summary JSON.

Usage: aggregate_agents.py <arm_dir> <total_wall_s> <arm_letter>
"""
import sys, json, glob, statistics
from pathlib import Path

arm_dir = Path(sys.argv[1])
total_wall = int(sys.argv[2])
arm = sys.argv[3]

results = []
for f in sorted(arm_dir.glob("rank_*/result.json")):
    try:
        results.append(json.loads(f.read_text()))
    except Exception as e:
        print(f"WARN: parse {f}: {e}", file=sys.stderr)

ok = [r for r in results if r.get("exit_code", -1) == 0]
walls = [float(r["wall_seconds"]) for r in ok]
answers = [r.get("answer") for r in ok]
io_bytes = [r.get("data_dir_bytes", 0) for r in ok]

def stats(xs):
    if not xs:
        return {"min": None, "max": None, "mean": None, "p50": None, "p95": None}
    return {
        "min": min(xs), "max": max(xs),
        "mean": statistics.mean(xs),
        "p50": statistics.median(xs),
        "p95": statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs),
    }

# Answer correctness — ground truth lives at the test's groundtruth.json
gt_count = None
gt_path = Path("/home/sislam6/clio-search/eval/agent_io_test/groundtruth.json")
if gt_path.exists():
    gt = json.loads(gt_path.read_text())
    gt_count = gt.get("count")

# Try integer-parse of answers
parsed_answers = []
for a in answers:
    try:
        parsed_answers.append(int(a))
    except Exception:
        pass
correct = sum(1 for a in parsed_answers if gt_count is not None and a == gt_count)

summary = {
    "arm": arm,
    "ranks_total": len(results),
    "ranks_ok": len(ok),
    "ranks_failed": len(results) - len(ok),
    "total_wall_s": total_wall,
    "wall_per_rank_s": stats(walls),
    "answer_data_bytes": stats(io_bytes),
    "ground_truth_count": gt_count,
    "answer_distribution": {str(a): parsed_answers.count(a) for a in set(parsed_answers)},
    "exact_match_count": correct,
    "exact_match_rate": (correct / len(ok)) if ok else 0.0,
}
print(json.dumps(summary, indent=2))
