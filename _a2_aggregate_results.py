"""Aggregate A2 ablation judge results.

Reads the 10 pairwise CSVs (5 chapters × {A2_vs_Hygiene, A2_vs_GG}), counts
final_choice = a / b / tie per chapter, computes decisive win rate + one-sided
sign-test p-value. Writes summary table to _a2_aggregate_results.log so Claude
can read it back.

Sign-test: under H0 (no preference), decisive choices are Bernoulli(0.5). P-value
for "A2 better than B" = P(X >= a_wins | n = a_wins + b_wins, p = 0.5).
"""
from __future__ import annotations
import sys
import csv
from math import comb

LOG = open("_a2_aggregate_results.log", "w", encoding="utf-8", buffering=1)
sys.stdout = LOG
sys.stderr = LOG


def one_sided_sign_test(a_wins: int, b_wins: int) -> float:
    """P(X >= a_wins | n = a_wins + b_wins, p=0.5).
    Returns the p-value for 'A is strictly preferred to B'."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = a_wins
    return sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def summarize(tag: str, chaps=(6, 7, 8, 9, 10)):
    rows = []
    for c in chaps:
        path = f"evaluation/llm_judge/pairwise_chap{c}_{tag}_deepseek-chat.csv"
        a = b = t = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fc = (row.get("final_choice") or "").strip().lower()
                if fc == "a": a += 1
                elif fc == "b": b += 1
                elif fc == "tie": t += 1
        n = a + b + t
        decisive = a + b
        a_pct = a / decisive * 100 if decisive else 0
        p_a_gt_b = one_sided_sign_test(a, b)
        p_b_gt_a = one_sided_sign_test(b, a)
        rows.append({
            "chap": c, "n": n, "a": a, "b": b, "tie": t,
            "decisive": decisive, "a_pct": a_pct,
            "p_A>B": p_a_gt_b, "p_B>A": p_b_gt_a,
        })

    # Aggregate row
    A = sum(r["a"] for r in rows)
    B = sum(r["b"] for r in rows)
    T = sum(r["tie"] for r in rows)
    N = A + B + T
    DEC = A + B
    A_PCT = A / DEC * 100 if DEC else 0
    P_A = one_sided_sign_test(A, B)
    P_B = one_sided_sign_test(B, A)
    rows.append({
        "chap": "TOTAL", "n": N, "a": A, "b": B, "tie": T,
        "decisive": DEC, "a_pct": A_PCT,
        "p_A>B": P_A, "p_B>A": P_B,
    })
    return rows


def render(title: str, rows: list[dict], a_label: str, b_label: str):
    print()
    print("=" * 78)
    print(f"  {title}")
    print(f"  A = {a_label}    B = {b_label}")
    print("=" * 78)
    print(f"{'Chap':>6} {'n':>5} {'A':>5} {'B':>5} {'tie':>5} {'dec':>5} "
          f"{'A%':>7} {'p(A>B)':>10} {'p(B>A)':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{str(r['chap']):>6} {r['n']:>5} {r['a']:>5} {r['b']:>5} {r['tie']:>5} "
              f"{r['decisive']:>5} {r['a_pct']:>6.1f}% "
              f"{r['p_A>B']:>10.4f} {r['p_B>A']:>10.4f}")


print("A2 ABLATION RESULTS  (run date: 2026-05-16)")
print(f"Pipeline: qwen3-8b-local + RAG + TM + Hygiene + A2 (fastcoref)")
print(f"Judge: deepseek-chat, two-call position-swap protocol")

a2_vs_hyg = summarize("A2_vs_Hygiene")
a2_vs_gg = summarize("A2_vs_GG")

render("A. A2 vs Hygiene (within-pipeline marginal lift)",
       a2_vs_hyg, a_label="a2_on", b_label="hygiene_on")

render("B. A2 vs Google Translate (residual gap)",
       a2_vs_gg, a_label="a2_on", b_label="GG")

# Quick comparison to baseline numbers from thesis_insights.md
print()
print("=" * 78)
print("  CONTEXT: prior baselines from thesis_insights.md (2026-05-15)")
print("=" * 78)
print("  TM_on vs TM_off (A3 alone):           52.1% decisive (p=0.26, NOT sig)")
print("  Hygiene vs TM_on (in-pipeline):       55.0% decisive (p=0.053, borderline)")
print("  Hygiene vs GG (final gap):            32.0% decisive (p<0.0001, GG wins)")

a2_hyg_tot = a2_vs_hyg[-1]
a2_gg_tot = a2_vs_gg[-1]
print()
print("=" * 78)
print("  THIS RUN — A2 marginal effect summary")
print("=" * 78)
print(f"  A2 vs Hygiene:  {a2_hyg_tot['a_pct']:.1f}% decisive  "
      f"(p={a2_hyg_tot['p_A>B']:.4f} A>B, p={a2_hyg_tot['p_B>A']:.4f} B>A)")
print(f"  A2 vs GG:       {a2_gg_tot['a_pct']:.1f}% decisive  "
      f"(p={a2_gg_tot['p_A>B']:.4f} A>B, p={a2_gg_tot['p_B>A']:.4f} B>A)")
print()
print("  Interpretation crib:")
print("    A2-vs-Hygiene >= 55% with p<0.05  => A2 adds significant marginal lift")
print("    A2-vs-Hygiene ~50%, p~0.50         => A2 is null (Track 1 prediction held)")
print("    A2-vs-GG > 32% (hygiene baseline)  => A2 closed some gap to GG")
print("    A2-vs-GG ~= 32%                    => A2 did not move the GG gap")
