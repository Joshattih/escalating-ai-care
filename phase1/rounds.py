"""
rounds.py  —  the feedback rounds (round 1 and round 2).

What one feedback round does, in order:
  1. open the previous round's log (round0 -> round1, round1 -> round2)
  2. for each statement, gather the 75 ratings (25 roles x 3 models), drop IDK, take the median
       - odd count: the middle value; even count with two different middles: keep both
  3. for each statement, build the reasons package:
       one reason from a doctor AT the median, one from the doctor FARTHEST ABOVE it,
       one from the doctor FARTHEST BELOW it, and one from an IDK doctor if any.
       When several doctors qualify for a slot, one is drawn with a seeded random draw
       (same seed -> same pick every time). Which model and role wrote each reason is saved.
  4. for every (model, role, statement), call ask() once with the feedback added to the prompt:
       the doctor's own previous rating, the median, and the package's reasons,
       LEAVING OUT any reason written by the doctor's own model and the doctor's own reason.
  5. the new answers are logged as the new round. Resumable like round0.py.

Nothing here changes the question, the scale, the role texts or the statements.
Not used in this script, on purpose: no probe, no subtraction, no challenge round.
"""

import argparse, csv, json, random, statistics, sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import STATEMENTS_CSV, LOGS, MODELS, ROLE_IDS, TEMPERATURE, MAX_OUTPUT_TOKENS, IDK
from roles import ROLE_TEXTS
import call_model
from call_model import ask

N_FEEDBACK_ROUNDS = 2        # step 7: fixed 2026-10-06
SEED = 20261006              # the seed for every random draw in this script; written here, not chosen later

# ---------------------------------------------------------------------------
# Reading a finished round
# ---------------------------------------------------------------------------
def load_round(run_dir, round_id):
    """Return {(model, role_id, statement_id): record} for the latest logged, non-error answer."""
    out = {}
    for line in open(Path(run_dir) / "calls.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("round") == round_id and r.get("status") in ("valid", "idk", "invalid"):
            out[(r["model"], r["role_id"], r["statement_id"])] = r
    return out

def median_of(values):
    """values: list of ints 1..6. Returns a list with one value, or two if the middles differ."""
    v = sorted(values)
    if not v:
        return []
    n = len(v)
    if n % 2 == 1:
        return [v[n // 2]]
    a, b = v[n // 2 - 1], v[n // 2]
    return [a] if a == b else [a, b]

def build_package(prev, statement_id, rng):
    """
    From the previous round's answers to one statement, build the median and the reasons package.
    Each reason is a dict: model, role_id, rating (or "IDK"), reason text, slot name.
    """
    rows = [r for (m, ro, s), r in prev.items() if s == statement_id and r["status"] in ("valid", "idk")]
    rated = [r for r in rows if r["status"] == "valid"]
    idks = [r for r in rows if r["status"] == "idk"]
    ratings = [int(r["answer"]) for r in rated]
    med = median_of(ratings)
    pkg = []
    def pick(cands, slot):
        cands = [c for c in cands if c.get("reason")]        # only doctors who gave a reason
        if cands:
            c = rng.choice(cands)                               # seeded draw among ties
            pkg.append({"slot": slot, "model": c["model"], "role_id": c["role_id"], "rating": c["answer"], "reason": c["reason"]})
    for mv in med:                                              # one reason per median value (one or two)
        pick([r for r in rated if int(r["answer"]) == mv], f"at the median ({mv})")
    if ratings and med:
        hi = max(ratings); lo = min(ratings)
        if hi > max(med): pick([r for r in rated if int(r["answer"]) == hi], f"farthest above ({hi})")
        if lo < min(med): pick([r for r in rated if int(r["answer"]) == lo], f"farthest below ({lo})")
    if idks:
        pick(idks, "said IDK")
    return {"statement_id": statement_id, "median": med, "n_rated": len(ratings), "n_idk": len(idks), "reasons": pkg}

# ---------------------------------------------------------------------------
# The feedback text added to the prompt
# ---------------------------------------------------------------------------
def feedback_text(own_prev, package, reader_model):
    med = package["median"]
    med_txt = ("between " + " and ".join(map(str, med))) if len(med) == 2 else str(med[0]) if med else "not available"
    lines = [f"Your rating in the previous round was {own_prev}.", f"The group median is {med_txt}."]
    shown = [p for p in package["reasons"] if p["model"] != reader_model]   # never the reader's own model
    if shown:
        lines.append("Other doctors said:")
        for p in shown:
            lines.append(f'- a doctor who rated {p["rating"]}: "{p["reason"]}"')
    lines.append("Same statement, same question, same scale. Rate again, with one sentence of reason.")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# One feedback round
# ---------------------------------------------------------------------------
def run_round(run_dir, prev_round, new_round, statements, role_ids, models, workers):
    prev = load_round(run_dir, prev_round)
    rng = random.Random(f"{SEED}-{new_round}")
    packages = {s["statement_id"]: build_package(prev, s["statement_id"], rng) for s in statements}
    # save the packages: this is the record of who influenced whom
    with open(Path(run_dir) / f"{new_round}_packages.json", "w", encoding="utf-8") as f:
        json.dump(packages, f, ensure_ascii=False, indent=1)
    done = {k for k, r in load_round(run_dir, new_round).items()}
    jobs = [(m, ro, s) for m in models for ro in role_ids for s in statements if (m, ro, s["statement_id"]) not in done]
    print(f"{new_round}: {len(jobs)} calls to make ({len(done)} already logged)")
    counts = {m: {"valid": 0, "idk": 0, "invalid": 0, "error": 0} for m in models}

    def one(job):
        m, ro, s = job
        own = prev.get((m, ro, s["statement_id"]))
        own_prev = own["answer"] if own and own["status"] in ("valid", "idk") else "not recorded"
        fb = feedback_text(own_prev, packages[s["statement_id"]], m)
        return ask(m, s["statement_id"], s["statement"], role_id=ro, role_text=ROLE_TEXTS[ro],
                   run_dir=run_dir, round_id=new_round, feedback=fb)

    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            rec = fut.result(); counts[rec["model"]][rec["status"]] += 1; finished += 1
            if finished % 100 == 0 or finished == len(jobs): print(f"  {finished}/{len(jobs)} done", flush=True)
    for m, c in counts.items():
        print(f"  {m:24s} valid {c['valid']:6d}  idk {c['idk']:5d}  invalid {c['invalid']:5d}  error {c['error']:5d}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="the round0 run folder to continue from")
    ap.add_argument("--statements", type=int, default=None)
    ap.add_argument("--roles", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    # accept the folder as typed, or relative to the logs folder, or by name alone
    for cand in (Path(a.run_dir), LOGS.parent / a.run_dir, LOGS / Path(a.run_dir).name):
        if (cand / "calls.jsonl").exists():
            a.run_dir = cand
            break
    else:
        sys.exit(f"no calls.jsonl found for {a.run_dir}; expected under {LOGS}")
    with open(STATEMENTS_CSV, newline="", encoding="utf-8") as f:
        statements = list(csv.DictReader(f))
    if a.statements: statements = statements[:a.statements]
    role_ids = ROLE_IDS[:a.roles] if a.roles else ROLE_IDS
    models = a.models or list(MODELS)
    prev = "round0"
    for k in range(1, N_FEEDBACK_ROUNDS + 1):
        new = f"round{k}"
        run_round(Path(a.run_dir), prev, new, statements, role_ids, models, a.workers)
        prev = new
    print("done:", a.run_dir)

if __name__ == "__main__":
    main()
