"""Validator for v4 multi-prompt + multi-dataset runs.

Reads result.json (which now has 'dataset' field) and scores against
per-(dataset, prompt_id) GT.
"""
import json, re, sys
from pathlib import Path

# GT loaded from per-dataset prompt grids
ARGO_GRID = json.load(open("/home/sislam6/clio-search/eval/agent_io_test/prompts_grid.json"))
ERA5_GRID = json.load(open("/tmp/era5_prompts_with_gt.json"))

ARGO_PROMPTS = {p["id"]: p for p in ARGO_GRID["prompts"]}
ERA5_PROMPTS = {p["id"]: p for p in ERA5_GRID}


def expected(dataset: str, arm: str, pid: int):
    p = (ARGO_PROMPTS if dataset == "argo" else ERA5_PROMPTS).get(pid)
    if p is None:
        return None
    if arm == "a":
        return p.get("gt_naive")
    if arm in ("b", "c"):
        return p.get("gt_clio")
    return None


def script_int(agent_log_path: Path):
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
        run_dirs = sorted(results_root.glob("v4_*"))

    rows = []
    for d in run_dirs:
        for rd in sorted(d.glob("arm_*/rank_*")):
            try:
                res = json.loads((rd / "result.json").read_text())
            except Exception:
                continue
            arm = res.get("arm")
            ds = res.get("dataset", "argo")
            pid = res.get("prompt_id", -1)
            pname = res.get("prompt_name", "?")
            wall = res.get("wall_seconds")
            ans = res.get("answer")
            try: aint = int(ans)
            except: aint = None
            si = script_int(rd / "agent.log")
            exp = expected(ds, arm, pid)
            rows.append({
                "run": d.name, "ds": ds, "arm": arm, "rank": int(rd.name.split("_")[-1]),
                "pid": pid, "pname": pname, "wall": wall,
                "parsed": ans, "script": si, "expected": exp,
                "correct": (si is not None and exp is not None and si == exp),
            })

    print(f"{'ds':<5} {'arm':<3} {'rk':>2} {'pid':>3} {'prompt':<26} {'wall':>6} {'parsed':>7} {'script':>7} {'exp':>4} {'ok':>3}")
    print("-" * 90)
    for r in sorted(rows, key=lambda r: (r["ds"], r["arm"], r["rank"])):
        wall = f"{r['wall']:.1f}" if isinstance(r['wall'], (int, float)) else "—"
        print(f"{r['ds']:<5} {r['arm']:<3} {r['rank']:>2} {r['pid']:>3} {r['pname']:<26} {wall:>6} "
              f"{str(r['parsed'])[:7]:>7} {str(r['script']):>7} {str(r['expected']):>4} "
              f"{'✓' if r['correct'] else '✗':>3}")

    print()
    print(f"\n{'ds':<5} {'arm':<3} {'prompt':<26} {'ranks':>5} {'correct':>8}  ans-distribution  expected")
    by = {}
    for r in rows:
        by.setdefault((r['ds'], r['arm'], r['pid']), []).append(r)
    for (ds, arm, pid), rs in sorted(by.items()):
        n = len(rs); k = sum(1 for r in rs if r['correct'])
        # answer histogram
        hist = {}
        for r in rs:
            hist[str(r['script'])] = hist.get(str(r['script']), 0) + 1
        hist_s = ", ".join(f"{a}×{c}" for a, c in sorted(hist.items()))
        pname = (ARGO_PROMPTS if ds == "argo" else ERA5_PROMPTS).get(pid, {}).get("name", "?")
        exp = expected(ds, arm, pid)
        print(f"{ds:<5} {arm:<3} {pname:<26} {n:>5} {k}/{n:<6} {hist_s:<25} {exp}")

    print()
    arm_tot = {}
    for r in rows:
        k = (r["ds"], r["arm"])
        arm_tot.setdefault(k, [0, 0])
        arm_tot[k][0] += 1
        if r['correct']:
            arm_tot[k][1] += 1
    for (ds, arm), (n, k) in sorted(arm_tot.items()):
        print(f"  {ds:<5} arm {arm}: {k}/{n} correct overall")

    out = Path("/tmp/validation_v4_report.json")
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
