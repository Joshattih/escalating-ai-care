"""
summarize.py — the simplest description of each round, one round at a time.

For every round (round0, round1, round2) in a run folder it can produce:
  --tables   one CSV per round: statement_id, median, count at 1..6, idk        (<round>_table.csv)
  --figures  one heatmap per round: 75 doctors (rows, grouped by model) x 257 statements
             (columns, in one fixed order: the round 0 median, then statement id)  (<round>_heatmap.png)
  --counts   three plain counts per round, printed to the terminal
With no flag it does all three. No API calls; reads only calls.jsonl in the run folder.

    python summarize.py logs/round0_2026-10-07_0433 --figures
"""
import argparse, csv, json, statistics, sys
from pathlib import Path
from collections import defaultdict, Counter

from config import LOGS, MODELS, ROLE_IDS, IDK, STATEMENTS_CSV

ROUNDS = ["round0", "round1", "round2"]
MODEL_ORDER = list(MODELS.keys())                     # the order the bands appear in, top to bottom
SHORT = {"gpt-4.1-mini": "GPT", "claude-haiku-4-5": "Claude", "gemini-2.5-flash-lite": "Gemini"}
SCALE_WORDS = {1: "1  must keep talking", 2: "2  keep talking", 3: "3  lean keep talking",
               4: "4  lean hand over", 5: "5  hand over", 6: "6  must hand over now"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def resolve_run_dir(arg):
    for cand in (Path(arg), LOGS.parent / arg, LOGS / Path(arg).name):
        if (cand / "calls.jsonl").exists():
            return cand
    sys.exit(f"no calls.jsonl found for {arg}; expected under {LOGS}")


def load(run_dir):
    """answers[round][statement_id][(model, role_id)] = int rating, or IDK (invalid/error rows are skipped)."""
    answers = {r: defaultdict(dict) for r in ROUNDS}
    for line in open(run_dir / "calls.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        if rec["round"] not in answers or rec["status"] not in ("valid", "idk"):
            continue
        val = IDK if rec["status"] == "idk" else int(rec["answer"])
        answers[rec["round"]][rec["statement_id"]][(rec["model"], rec["role_id"])] = val
    return answers


def median_of(values):
    """One middle value, or the two middle values when the count is even (never averaged)."""
    vals = sorted(values)
    if not vals:
        return []
    n = len(vals)
    return [vals[n // 2]] if n % 2 else sorted(set([vals[n // 2 - 1], vals[n // 2]]))


def statement_order(answers, by="id"):
    """Column order for the figures. Default: plain statement order, as in statements.csv (01-003 first).
    by="median": by round 0 median (low to high), then statement id; kept as an option for later pictures."""
    if by == "id":
        ids = [r["statement_id"] for r in csv.DictReader(open(STATEMENTS_CSV, encoding="utf-8"))]
        return [sid for sid in ids if sid in answers["round0"]]
    rows = []
    for sid, d in answers["round0"].items():
        nums = [v for v in d.values() if v != IDK]
        med = median_of(nums)
        key = (sum(med) / len(med)) if med else 99      # a two-value median sorts between its two values
        rows.append((key, sid))
    return [sid for _, sid in sorted(rows)]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_table(answers, rnd, order, run_dir):
    out = run_dir / f"{rnd}_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["statement_id", "median", "n1", "n2", "n3", "n4", "n5", "n6", "idk", "n_rated"])
        for sid in order:
            d = answers[rnd].get(sid, {})
            nums = [v for v in d.values() if v != IDK]
            c = Counter(nums)
            med = median_of(nums)
            w.writerow([sid, " and ".join(map(str, med)), *[c[v] for v in range(1, 7)],
                        sum(1 for v in d.values() if v == IDK), len(nums)])
    return out


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
def counts(answers, rnd):
    one = two = within1 = any_idk = 0
    for sid, d in answers[rnd].items():
        nums = [v for v in d.values() if v != IDK]
        med = median_of(nums)
        if len(med) == 1: one += 1
        elif len(med) == 2: two += 1
        lo, hi = (min(med), max(med)) if med else (0, 0)
        if nums and all(lo - 1 <= v <= hi + 1 for v in nums): within1 += 1
        if any(v == IDK for v in d.values()): any_idk += 1
    n = len(answers[rnd])
    return (f"{rnd}: {n} statements | median is one value: {one}, two values: {two} | "
            f"every doctor within one point of the median: {within1} | at least one IDK: {any_idk}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def draw_heatmap(answers, rnd, order, run_dir, axis_text=True, legend=True, figsize=(26, 10)):
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    rows = [(m, r) for m in MODEL_ORDER for r in ROLE_IDS]          # 75 rows, grouped by model
    grid = np.full((len(rows), len(order)), np.nan)
    for j, sid in enumerate(order):
        d = answers[rnd].get(sid, {})
        for i, key in enumerate(rows):
            v = d.get(key)
            if v is None: continue
            grid[i, j] = 0 if v == IDK else v                        # 0 = IDK, drawn grey

    colors = ["#8c8c8c", "#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"]  # IDK, 1..6
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    n_roles = len(ROLE_IDS)
    for b in range(1, len(MODEL_ORDER)):                             # thin lines between the three bands
        ax.axhline(b * n_roles - 0.5, color="white", linewidth=3)
    ax.set_yticks([b * n_roles + n_roles / 2 - 0.5 for b in range(len(MODEL_ORDER))])
    ax.set_yticklabels([f"{SHORT[m]}\n25 doctors" for m in MODEL_ORDER], fontsize=18, fontweight="bold")
    ticks = [0] + [k for k in range(24, len(order), 25) if k < len(order) - 12] + [len(order) - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{k+1}\n{order[k]}" for k in ticks], fontsize=13, fontweight="bold")
    if axis_text:
        ax.set_xlabel("The 257 statements in the order of the statements table. Tick labels: position, then statement id.",
                      fontsize=18, fontweight="bold", labelpad=14)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#9a9a9a"); sp.set_linewidth(1.5)

    if legend:
        handles = [Patch(color=colors[v], label=SCALE_WORDS[v]) for v in range(1, 7)] + [Patch(color=colors[0], label="IDK  I don't know")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  fontsize=16, handlelength=1.6, handleheight=1.6, labelspacing=0.9)
    out = run_dir / f"{rnd}_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out



def draw_band(answers, rnd, model, order, run_dir, axis_text=True, legend=True, figsize=(26, 9)):
    """One model's 25 doctors in one round: rows D1..D25, columns the 257 statements, fully labelled."""
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    grid = np.full((len(ROLE_IDS), len(order)), np.nan)
    for j, sid in enumerate(order):
        d = answers[rnd].get(sid, {})
        for i, r in enumerate(ROLE_IDS):
            v = d.get((model, r))
            if v is not None:
                grid[i, j] = 0 if v == IDK else v
    colors = ["#8c8c8c", "#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(ROLE_IDS)))
    ax.set_yticklabels(ROLE_IDS, fontsize=14, fontweight="bold")
    if axis_text: ax.set_ylabel(f"The 25 {SHORT[model]} doctors", fontsize=20, fontweight="bold", labelpad=12)
    ticks = [0] + [t for t in range(24, len(order), 25) if t < len(order) - 12] + [len(order) - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t+1}\n{order[t]}" for t in ticks], fontsize=13, fontweight="bold")
    if axis_text:
        ax.set_xlabel("The 257 statements, left to right from the lowest group median in round 0 (keep talking) to the highest (hand over now)."
                      "  Tick labels: position, then statement id.", fontsize=17, fontweight="bold", labelpad=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#9a9a9a"); sp.set_linewidth(1.5)
    if legend:
        handles = [Patch(color=colors[v], label=SCALE_WORDS[v]) for v in range(1, 7)] + [Patch(color=colors[0], label="IDK  I don't know")]
        ax.legend(handles=handles, title="one square = one doctor's rating\nof one statement", title_fontsize=15,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=16, handlelength=1.6, handleheight=1.6, labelspacing=0.9)
    out = run_dir / f"{rnd}_{SHORT[model].lower()}_band.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--band", nargs=2, metavar=("ROUND", "MODEL"), help="one model, one round, fully labelled")
    ap.add_argument("--order", choices=["id", "median"], default="id", help="column order: statement id (default) or round 0 median")
    ap.add_argument("--no-legend", action="store_true", help="leave the colour key off the picture (to add it as slide objects)")
    ap.add_argument("--only", nargs="+", choices=ROUNDS, help="limit to these rounds")
    ap.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"), default=[26, 9], help="figure size in inches")
    ap.add_argument("--no-axis-text", action="store_true", help="leave the axis titles off the picture (to add them as slide text)")
    a = ap.parse_args()
    do_all = not (a.tables or a.figures or a.counts)
    run_dir = resolve_run_dir(a.run_dir)
    answers = load(run_dir)
    order = statement_order(answers, a.order)
    if a.band:
        print("figure: ", draw_band(answers, a.band[0], a.band[1], order, run_dir, axis_text=not a.no_axis_text, legend=not a.no_legend, figsize=tuple(a.figsize))); return
    for rnd in (a.only or ROUNDS):
        if not answers[rnd]:
            print(f"{rnd}: no answers in the log, skipped"); continue
        if do_all or a.tables:  print("table:  ", write_table(answers, rnd, order, run_dir))
        if do_all or a.figures: print("figure: ", draw_heatmap(answers, rnd, order, run_dir, axis_text=not a.no_axis_text, legend=not a.no_legend, figsize=tuple(a.figsize)))
        if do_all or a.counts:  print(counts(answers, rnd))


if __name__ == "__main__":
    main()
