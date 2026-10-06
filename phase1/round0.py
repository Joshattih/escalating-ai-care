"""
round0.py  —  Round 0, private (step B).

Every model plays every role; every role rates every statement once, alone.
One call per (model, role, statement). No doctor sees any other doctor, any median,
or any reason. The reason each doctor gives is logged and not read by this script.

Usage (run from the phase1 folder):
    python round0.py                      full run: 3 models x 25 roles x 257 statements = 19,275 calls
    python round0.py --statements 5 --roles 3     test run: 3 x 3 x 5 = 45 calls
    python round0.py --resume logs/round0_2026-10-06_1502   continue a stopped run

What it writes: one JSON line per call in <run_dir>/calls.jsonl (through call_model.ask),
plus run_info.json with the settings used. Nothing is ever overwritten; a new run gets a
new folder named with the date and time.
"""

import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import STATEMENTS_CSV, LOGS, MODELS, ROLE_IDS, TEMPERATURE, MAX_OUTPUT_TOKENS
from roles import ROLE_TEXTS
from call_model import ask

def load_statements(limit=None):
    """The 257 statements, in file order. limit keeps only the first N (test runs)."""
    with open(STATEMENTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows

def already_done(run_dir):
    """Set of (model, role_id, statement_id) with a logged, non-error answer. Used to resume."""
    done = set()
    p = Path(run_dir) / "calls.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("round") == "round0" and r.get("status") in ("valid", "idk", "invalid"):
                done.add((r["model"], r["role_id"], r["statement_id"]))
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", type=int, default=None, help="use only the first N statements (test runs)")
    ap.add_argument("--roles", type=int, default=None, help="use only the first N roles (test runs)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names (default: all three)")
    ap.add_argument("--workers", type=int, default=8, help="calls in flight at once")
    ap.add_argument("--resume", default=None, help="existing run folder to continue")
    a = ap.parse_args()

    statements = load_statements(a.statements)
    role_ids = ROLE_IDS[:a.roles] if a.roles else ROLE_IDS
    models = a.models or list(MODELS)

    # one folder per run; a resume reuses the given folder
    run_dir = Path(a.resume) if a.resume else LOGS / f"round0_{datetime.now():%Y-%m-%d_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "round": "round0", "models": models, "roles": role_ids, "n_statements": len(statements),
        "temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS, "started": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    done = already_done(run_dir)
    jobs = [(m, r, s) for m in models for r in role_ids for s in statements if (m, r, s["statement_id"]) not in done]
    total = len(models) * len(role_ids) * len(statements)
    print(f"run folder: {run_dir}")
    print(f"planned {total} calls; {len(done)} already logged; {len(jobs)} to make")

    counts = {m: {"valid": 0, "idk": 0, "invalid": 0, "error": 0} for m in models}
    def one(job):
        m, r, s = job
        return ask(m, s["statement_id"], s["statement"], role_id=r, role_text=ROLE_TEXTS[r],
                   run_dir=run_dir, round_id="round0")

    finished = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            counts[rec["model"]][rec["status"]] += 1
            finished += 1
            if finished % 100 == 0 or finished == len(jobs):
                print(f"  {finished}/{len(jobs)} done", flush=True)

    print("\nanswers per model (this session):")
    for m, c in counts.items():
        print(f"  {m:24s} valid {c['valid']:6d}  idk {c['idk']:5d}  invalid {c['invalid']:5d}  error {c['error']:5d}")
    print(f"\nlog: {run_dir / 'calls.jsonl'}")

if __name__ == "__main__":
    main()
