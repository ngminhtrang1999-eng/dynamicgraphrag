"""Score the expanded human-evaluation kit: inter-rater agreement + judge match.

Consumes:
    evaluation/human_eval/human_eval_key.csv
    evaluation/human_eval/human_eval_rater_*.csv   (one per rater; `choice` filled)

Reports, per comparison and overall:
    * pairwise Cohen's kappa between every rater pair (3-way label A/B/tie)
    * Fleiss' kappa across all raters when >=3 raters are present
    * raw percent agreement
    * human decisive-win rate (candidate vs GG) vs the LLM judge's decisive-win
      rate on the same items -- the aggregate-level validation the paper claims
    * human-vs-judge decisive agreement (the cross-check the reviewer asked for)

Pure-stdlib kappa implementations (no sklearn/statsmodels dependency) so it runs
in any environment.
"""

import csv
import glob
import itertools
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
HE = ROOT / "evaluation" / "human_eval"
KEY = HE / "human_eval_key.csv"
LABELS = ["A", "B", "tie"]


def norm(choice):
    c = (choice or "").strip().lower()
    if c in {"a", "b", "tie"}:
        return {"a": "A", "b": "B", "tie": "tie"}[c]
    return None


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for two raters over paired categorical labels."""
    pairs = [(x, y) for x, y in zip(labels_a, labels_b) if x and y]
    n = len(pairs)
    if n == 0:
        return float("nan"), 0
    po = sum(1 for x, y in pairs if x == y) / n
    cnt_a = defaultdict(int)
    cnt_b = defaultdict(int)
    for x, y in pairs:
        cnt_a[x] += 1
        cnt_b[y] += 1
    pe = sum((cnt_a[l] / n) * (cnt_b[l] / n) for l in LABELS)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return kappa, n


def fleiss_kappa(item_label_lists):
    """Fleiss' kappa. item_label_lists: list of per-item [label,...] (>=2 raters)."""
    items = [labs for labs in item_label_lists if len(labs) >= 2]
    if not items:
        return float("nan"), 0
    # require constant rater count for the canonical formula
    counts = {len(x) for x in items}
    if len(counts) != 1:
        # fall back: keep items whose rater count == the mode
        from collections import Counter
        mode_n = Counter(len(x) for x in items).most_common(1)[0][0]
        items = [x for x in items if len(x) == mode_n]
    n_raters = len(items[0])
    N = len(items)
    if N == 0 or n_raters < 2:
        return float("nan"), 0

    n_ij = []
    for labs in items:
        row = {l: 0 for l in LABELS}
        for l in labs:
            row[l] += 1
        n_ij.append(row)

    p_j = {l: sum(row[l] for row in n_ij) / (N * n_raters) for l in LABELS}
    P_i = []
    for row in n_ij:
        s = sum(row[l] * (row[l] - 1) for l in LABELS)
        P_i.append(s / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / N
    Pe_bar = sum(p_j[l] ** 2 for l in LABELS)
    kappa = (P_bar - Pe_bar) / (1 - Pe_bar) if (1 - Pe_bar) else float("nan")
    return kappa, N


def load_key():
    with open(KEY, encoding="utf-8-sig") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def load_raters():
    raters = {}
    for path in sorted(glob.glob(str(HE / "human_eval_rater_*.csv"))):
        name = Path(path).stem.replace("human_eval_rater_", "")
        with open(path, encoding="utf-8-sig") as f:
            choices = {r["item_id"]: norm(r.get("choice")) for r in csv.DictReader(f)}
        n_filled = sum(1 for v in choices.values() if v)
        if n_filled == 0:
            print(f"  (skipping {name}: no choices filled)")
            continue
        raters[name] = choices
        print(f"  loaded rater '{name}': {n_filled} choices")
    return raters


def human_choice_to_system(choice, keyrow):
    """Map a rater's A/B/tie to candidate-vs-GG convention (a=cand, b=GG, tie)."""
    if choice == "tie":
        return "tie"
    side_is = keyrow["A_is"] if choice == "A" else keyrow["B_is"]
    return "a" if side_is == "CAND" else "b"


def decisive_rate(verdicts):
    a = verdicts.count("a")
    b = verdicts.count("b")
    dec = a + b
    return (a / dec * 100 if dec else float("nan")), a, b, verdicts.count("tie"), dec


def main():
    if not KEY.exists():
        raise SystemExit(f"missing {KEY}; run build_human_eval_kit.py first")
    key = load_key()
    print("Loading raters...")
    raters = load_raters()
    if not raters:
        raise SystemExit("no rater files with filled choices found "
                         "(expected evaluation/human_eval/human_eval_rater_*.csv)")

    names = sorted(raters)
    item_ids = sorted(key, key=lambda x: int(x))
    comparisons = sorted({key[i]["comparison"] for i in item_ids})

    # ---- inter-rater agreement (overall) ----
    print("\n=== Inter-rater agreement (all items) ===")
    if len(names) >= 2:
        for a, b in itertools.combinations(names, 2):
            la = [raters[a].get(i) for i in item_ids]
            lb = [raters[b].get(i) for i in item_ids]
            k, n = cohen_kappa(la, lb)
            po = sum(1 for x, y in zip(la, lb) if x and y and x == y)
            tot = sum(1 for x, y in zip(la, lb) if x and y)
            print(f"  Cohen kappa {a} vs {b}: {k:.3f} "
                  f"(raw agree {po}/{tot} = {po/tot*100:.1f}%)" if tot else
                  f"  Cohen kappa {a} vs {b}: n/a")
        if len(names) >= 3:
            ill = []
            for i in item_ids:
                labs = [raters[nm].get(i) for nm in names]
                labs = [x for x in labs if x]
                ill.append(labs)
            fk, N = fleiss_kappa(ill)
            print(f"  Fleiss kappa ({len(names)} raters, {N} complete items): {fk:.3f}")
    else:
        print("  only one rater -- no inter-rater agreement to compute")

    # ---- per-comparison: human vs judge decisive rate ----
    print("\n=== Human vs LLM-judge decisive-win rate (candidate vs GG) ===")
    header = f"{'comparison':<22} {'rater':<10} {'human A%':>9} {'judge A%':>9} {'h-vs-j agree':>13}"
    print(header)
    print("-" * len(header))
    for comp in comparisons:
        comp_items = [i for i in item_ids if key[i]["comparison"] == comp]
        judge_v = [key[i]["judge_verdict"] for i in comp_items]
        jrate, *_ , jdec = decisive_rate(judge_v)
        for nm in names:
            hv, jv_match, tot_dec = [], 0, 0
            for i in comp_items:
                ch = raters[nm].get(i)
                if not ch:
                    continue
                hsys = human_choice_to_system(ch, key[i])
                hv.append(hsys)
                jv = key[i]["judge_verdict"]
                if hsys in {"a", "b"} and jv in {"a", "b"}:
                    tot_dec += 1
                    if hsys == jv:
                        jv_match += 1
            hrate, *_ = decisive_rate(hv)
            agree = f"{jv_match}/{tot_dec}={jv_match/tot_dec*100:.0f}%" if tot_dec else "n/a"
            print(f"{comp:<22} {nm:<10} {hrate:>8.1f}% {jrate:>8.1f}% {agree:>13}")
    print("\nNote: 'human A%' and 'judge A%' are decisive-win rates of the "
          "candidate system over GG; closeness validates the judge at the "
          "aggregate level. 'h-vs-j agree' is per-item decisive agreement.")


if __name__ == "__main__":
    main()
