"""Aggregate new Gemini judgments + judge-agreement analysis.

Usage:
    python analyze_new_judgments.py aggregate   # win rates for all gemini35flash pairwise CSVs
    python analyze_new_judgments.py agreement   # old deepseek vs new gemini verdicts on rejudge sample
"""

import csv
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

OUT = Path(__file__).parent / "evaluation" / "llm_judge"


def sign_test_p(k, n):
    """One-sided sign test: P(X >= k), X ~ Bin(n, 0.5)."""
    if n == 0:
        return float("nan")
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def aggregate():
    groups = defaultdict(list)
    for p in sorted(OUT.glob("pairwise_chap*_gemini35flash.csv")):
        # pairwise_chap<N>_<SLUG>_vs_GG_gemini35flash.csv
        slug = p.stem.split("_", 2)[2].rsplit("_gemini35flash", 1)[0]
        groups[slug].extend(read(p))
    print(f"{'comparison':<28} {'n':>5} {'n_dec':>6} {'A%':>7} {'p':>10}")
    for slug, rows in sorted(groups.items()):
        a = sum(1 for r in rows if r["final_choice"] == "a")
        b = sum(1 for r in rows if r["final_choice"] == "b")
        n_dec = a + b
        pct = 100 * a / n_dec if n_dec else float("nan")
        p = sign_test_p(max(a, b), n_dec)
        print(f"{slug:<28} {len(rows):>5} {n_dec:>6} {pct:>6.1f}% {p:>10.2g}")


def agreement():
    print(f"{'comparison':<18} {'n':>4} {'exact%':>7} {'dec-dec%':>9} "
          f"{'oldA%':>7} {'newA%':>7}")
    for newp in sorted(OUT.glob("rejudge_*_gemini35flash.csv")):
        name = newp.stem.replace("rejudge_", "").replace("_gemini35flash", "")
        new = {(r["chapter"], r["sentence_id"]): r["final_choice"] for r in read(newp)}
        old = {}
        for oldp in OUT.glob("pairwise_*deepseek-chat.csv"):
            for r in read(oldp):
                old[(r["chapter"], r["sentence_id"], oldp.name)] = r["final_choice"]
        # match old verdicts by comparison-specific file pattern
        pat_map = {
            "TMon_vs_TMoff": "TMon_vs_TMoff", "Hygiene_vs_TMon": "Hygiene_vs_TMon",
            "A2_vs_Hygiene": "A2_vs_Hygiene", "A2_vs_GG": "A2_vs_GG",
            "RAGBIG_vs_GG": "RAGBIG_vs_GG",
        }
        pat = pat_map[name]
        old_match = {}
        for (ch, sid, fname), v in old.items():
            if pat in fname:
                old_match[(ch, sid)] = v
        pairs = [(old_match.get(k), v) for k, v in new.items() if k in old_match]
        n = len(pairs)
        if not n:
            print(f"{name:<18} {0:>4}  (no overlap)")
            continue
        exact = sum(1 for o, v in pairs if o == v) / n
        both_dec = [(o, v) for o, v in pairs if o in ("a", "b") and v in ("a", "b")]
        decdec = (sum(1 for o, v in both_dec if o == v) / len(both_dec)) if both_dec else float("nan")
        old_a = [o for o, _ in pairs if o in ("a", "b")]
        new_a = [v for _, v in pairs if v in ("a", "b")]
        old_pct = 100 * old_a.count("a") / len(old_a) if old_a else float("nan")
        new_pct = 100 * new_a.count("a") / len(new_a) if new_a else float("nan")
        print(f"{name:<18} {n:>4} {exact * 100:>6.1f}% {decdec * 100:>8.1f}% "
              f"{old_pct:>6.1f}% {new_pct:>6.1f}%")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    {"aggregate": aggregate, "agreement": agreement}[mode]()
