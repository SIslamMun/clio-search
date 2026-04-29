"""Validator for v3 multi-prompt runs.

Per-rank result.json now includes 'prompt_id' and 'prompt_name'. We score each
rank against the ground truth for *its* prompt (not a single arm-level GT).

GT comes from /home/sislam6/clio-search/eval/agent_io_test/prompts_grid.json
which has both gt_naive (literal surface-only answer) and gt_clio (any-depth
answer from the structured query).
"""
import json, re, sys
from pathlib import Path

GRID = json.load(open("/home/sislam6/clio-search/eval/agent_io_test/prompts_grid.json"))
PROMPTS = {p["id"]: p for p in GRID["prompts"]}

ARM_LABEL = {
    "a": "Naive netCDF4 scan (LLM Python)",
    "b": "CLIO + DuckDB only",
    "c": "CLIO + IOWarp at query",
    "d": "Federated CLIO across 100 shards",
}


def expected(arm: str, prompt_id: int) -> int | None:
    """GT depends on arm: A targets surface-literal, B/C target CLIO any-depth."""
    p = PROMPTS.get(prompt_id)
    if p is None:
        return None
    if arm == "a":
        return p["gt_naive"]
    if arm in ("b", "c"):
        return p["gt_clio"]
    return None  # D: single prompt, separate GT


def script_int(agent_log_path: Path) -> int | None:
    if not agent_log_path.exists():
        return None
    txt = agent_log_path.read_text(errors="replace")
    m = re.search(r"===== BEGIN OPENCODE RAW OUTPUT =====(.*?)===== END OPENCODE RAW OUTPUT",
                  txt, flags=re.DOTALL)
    block = m.group(1) if m else txt
    block = re.sub(r"\x1b\[[0-9;]*m", "", block)
    nums = re.findall(r"^\s*(\d+)\s*$", block, flags=re.MULTILINE)
    return int(nums[-1]) if nums else None


def main():
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        results_root = Path("/home/sislam6/clio-search/eval/agent_io_test/results")
        run_dirs = sorted(results_root.glob("v3_*"))

    rows = []
    for d in run_dirs:
        for rd in sorted(d.glob("arm_*/rank_*")):
            arm = rd.parent.name.replace("arm_", "")
            try:
                res = json.loads((rd / "result.json").read_text())
            except Exception:
                continue
            try:
                meta = json.loads((rd / "meta.json").read_text())
            except Exception:
                meta = {}
            rank = int(rd.name.replace("rank_", ""))
            pid = res.get("prompt_id", meta.get("prompt_id", -1))
            pname = res.get("prompt_name", meta.get("prompt_name", "?"))
            wall = res.get("wall_seconds")
            answer = res.get("answer")
            try:
                aint = int(answer)
            except (TypeError, ValueError):
                aint = None
            si = script_int(rd / "agent.log")
            exp = expected(arm, pid)
            rows.append({
                "run": d.name, "arm": arm, "rank": rank,
                "prompt_id": pid, "prompt_name": pname,
                "wall": wall, "parsed": answer, "script": si,
                "expected": exp,
                "correct": (si is not None and exp is not None and si == exp),
            })

    print(f"{'arm':<3} {'rk':>2} {'pid':>3} {'prompt':<24} {'wall':>6} {'parsed':>6} {'script':>6} {'exp':>4} {'ok':>3}")
    print("-" * 80)
    for r in rows:
        wall = f"{r['wall']:.1f}" if isinstance(r['wall'], (int, float)) else "—"
        print(f"{r['arm']:<3} {r['rank']:>2} {r['prompt_id']:>3} {r['prompt_name']:<24} {wall:>6} "
              f"{str(r['parsed'])[:6]:>6} {str(r['script']):>6} {str(r['expected']):>4} "
              f"{'✓' if r['correct'] else '✗':>3}")

    print()
    by = {}
    for r in rows:
        by.setdefault((r['arm'], r['prompt_id']), []).append(r)
    print(f"\n{'arm':<3} {'prompt':<24} {'ranks':>5} {'correct':>8}  expected")
    for (arm, pid), rs in sorted(by.items()):
        n = len(rs); k = sum(1 for r in rs if r['correct'])
        pname = PROMPTS.get(pid, {}).get("name", "?")
        exp = expected(arm, pid)
        print(f"{arm:<3} {pname:<24} {n:>5} {k}/{n:<6} {exp}")

    # Per-arm overall
    print()
    arm_tot = {}
    for r in rows:
        arm_tot.setdefault(r['arm'], [0, 0])
        arm_tot[r['arm']][0] += 1
        if r['correct']:
            arm_tot[r['arm']][1] += 1
    for arm, (n, k) in sorted(arm_tot.items()):
        print(f"  arm {arm} ({ARM_LABEL[arm]:<35}) {k}/{n} correct")

    out = Path("/tmp/validation_v3_report.json")
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
