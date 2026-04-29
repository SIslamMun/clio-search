#!/usr/bin/env python3
"""Validate agent results: cross-check parsed answer vs script-printed value vs ground truth.

For each rank workspace:
  - Extract what the agent's script ACTUALLY printed (between BEGIN/END markers in agent.log)
  - Compare to result.json's parsed answer
  - Compare to ground truth for that arm
  - Report per-rank pass/fail with the evidence

Usage: python3 validate_results.py [run_dir1 run_dir2 ...]
       (no args -> auto-discover today's v2_* dirs)
"""
import json
import re
import sys
from pathlib import Path

# Ground truth per arm — UNIFIED 200-blob corpus
# (rebuilt 2026-04-29; blobs_unified.json + test_index_unified.duckdb)
GT = {
    # Arm A: naive netCDF4 surface-temp scan over 200 unified Argo profiles.
    #   groundtruth_unified.json: 0/200 have surface temp > 30 degC.
    "a": 0,
    # Arm B: CLIO search_scientific over test_index_unified.duckdb.
    #   Numeric range temperature >= 86 degF (>= 30 degC), ANY depth.
    #   Verified via direct query against the index.
    "b": 9,
    # Arm C: search_scientific_with_content (same query, same index).
    #   Same numeric range -> same 9 chunk hits.
    "c": 9,
    # Arm D: federated query top-K across 100 Argo shards (separate corpus).
    #   federated_query.py default top_k=50; UAN smoke test returned 50.
    "d": 50,
}

ARM_LABEL = {
    "a": "Naive netCDF4 scan",
    "b": "CLIO + DuckDB only",
    "c": "CLIO + IOWarp at query",
    "d": "Federated CLIO across 100 shards",
}


def extract_script_output(agent_log_path: Path) -> str:
    """Pull text strictly between BEGIN/END opencode markers."""
    if not agent_log_path.exists():
        return ""
    txt = agent_log_path.read_text(errors="replace")
    m = re.search(
        r"===== BEGIN OPENCODE RAW OUTPUT =====(.*?)===== END OPENCODE RAW OUTPUT",
        txt, flags=re.DOTALL,
    )
    return m.group(1) if m else txt


def extract_last_int(text: str) -> "int | None":
    """Last integer that appears on its own line (best signal of script output)."""
    # Strip ANSI escapes
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    nums = re.findall(r"^\s*(\d+)\s*$", clean, flags=re.MULTILINE)
    if nums:
        return int(nums[-1])
    # Fallback: any integer
    nums = re.findall(r"\b(\d+)\b", clean)
    return int(nums[-1]) if nums else None


def check_iowarp_evidence(agent_log_path: Path) -> int:
    """Count lines in agent.log that mention iowarp/CLIO with-content APIs."""
    if not agent_log_path.exists():
        return 0
    txt = agent_log_path.read_text(errors="replace")
    return sum(
        1 for line in txt.splitlines()
        if re.search(r"fetch_blob_content|search_scientific_with_content|Tag\.GetBlob|cte\.Tag", line)
    )


def validate_rank(rank_dir: Path) -> dict:
    arm = rank_dir.parent.name.replace("arm_", "")
    rank = rank_dir.name.replace("rank_", "")

    result_json_path = rank_dir / "result.json"
    agent_log_path = rank_dir / "agent.log"
    prompt_path = rank_dir / "prompt.txt"

    record = {
        "arm": arm,
        "rank": rank,
        "label": ARM_LABEL.get(arm, "?"),
        "rank_dir": str(rank_dir),
        "ground_truth": GT.get(arm),
    }

    # Parse result.json
    try:
        result = json.loads(result_json_path.read_text())
        record["host"] = result.get("host")
        record["wall_seconds"] = result.get("wall_seconds")
        record["exit_code"] = result.get("exit_code")
        record["data_dir_bytes"] = result.get("data_dir_bytes")
        record["parsed_answer"] = result.get("answer", result.get("answers"))
    except Exception as e:
        record["result_json_error"] = str(e)
        record["parsed_answer"] = None

    # Independent extraction from agent.log
    script_output = extract_script_output(agent_log_path)
    record["script_printed_int"] = extract_last_int(script_output)
    record["agent_log_size"] = agent_log_path.stat().st_size if agent_log_path.exists() else 0
    record["prompt_size"] = prompt_path.stat().st_size if prompt_path.exists() else 0
    record["iowarp_evidence_lines"] = check_iowarp_evidence(agent_log_path)

    # Validation logic
    parsed = record.get("parsed_answer")
    script = record.get("script_printed_int")
    gt = record.get("ground_truth")

    # Coerce parsed for comparison
    parsed_int = None
    if isinstance(parsed, str):
        try: parsed_int = int(parsed)
        except (ValueError, TypeError): pass
    elif isinstance(parsed, int):
        parsed_int = parsed

    record["parsed_matches_script"] = (parsed_int == script) if parsed_int is not None and script is not None else False
    if gt is not None:
        record["script_correct"] = (script == gt)
        record["parsed_correct"] = (parsed_int == gt)
    else:
        record["script_correct"] = None
        record["parsed_correct"] = None

    return record


def main():
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        # Auto-discover today's v2 dirs
        results_root = Path("/home/sislam6/clio-search/eval/agent_io_test/results")
        run_dirs = sorted(results_root.glob("v2_20260429_*"))

    print(f"Validating {len(run_dirs)} run dirs", flush=True)
    all_records = []
    for d in run_dirs:
        for rank_dir in sorted(d.glob("arm_*/rank_*")):
            rec = validate_rank(rank_dir)
            all_records.append(rec)

    # Print table
    print()
    print(f"{'run':<26} {'arm':<3} {'rank':<5} {'wall':>6} {'parsed':>7} {'script':>7} {'GT':>5} {'match':>6} {'correct':>8} {'iowarp':>7}")
    print("-" * 100)
    for r in all_records:
        run_name = Path(r["rank_dir"]).parents[1].name
        wall = f"{r.get('wall_seconds', 0):.1f}" if r.get('wall_seconds') else "—"
        parsed = str(r.get("parsed_answer"))[:7]
        script = str(r.get("script_printed_int"))[:7]
        gt = str(r.get("ground_truth")) if r.get("ground_truth") is not None else "—"
        match = "✓" if r.get("parsed_matches_script") else "✗"
        correct = "✓" if r.get("script_correct") else ("—" if r.get("script_correct") is None else "✗")
        iowarp = r.get("iowarp_evidence_lines", 0)
        print(f"{run_name:<26} {r['arm']:<3} {r['rank']:<5} {wall:>6} {parsed:>7} {script:>7} {gt:>5} {match:>6} {correct:>8} {iowarp:>7}")

    # Summary
    print()
    print("Summary by arm (using script-printed values):")
    by_arm = {}
    for r in all_records:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm in sorted(by_arm):
        ranks = by_arm[arm]
        gt = GT.get(arm)
        if gt is not None:
            correct = sum(1 for r in ranks if r.get("script_correct"))
            print(f"  {arm} ({ARM_LABEL[arm]:<35}) GT={gt}  correct={correct}/{len(ranks)}")
        else:
            answers = [r.get("script_printed_int") for r in ranks]
            print(f"  {arm} ({ARM_LABEL[arm]:<35}) GT=None  answers={answers}")

    # Save full report
    out = Path("/tmp/validation_report.json")
    out.write_text(json.dumps(all_records, indent=2, default=str))
    print(f"\nFull report saved: {out}")


if __name__ == "__main__":
    main()
