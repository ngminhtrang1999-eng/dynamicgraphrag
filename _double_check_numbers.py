"""Definitive aggregation of all judge results for paper Section V.
Re-reads every CSV from disk to ensure published numbers match raw data.
"""
import pandas as pd
from scipy.stats import binomtest
from pathlib import Path

JUDGE_DIR = Path("evaluation/llm_judge")


def agg(files, label):
    a_tot, b_tot, t_tot, n_tot = 0, 0, 0, 0
    per_chap = []
    for f in files:
        if not f.exists():
            per_chap.append((f.name, "MISSING"))
            continue
        df = pd.read_csv(f, encoding="utf-8-sig")
        a = (df["final_choice"] == "a").sum()
        b = (df["final_choice"] == "b").sum()
        t = (df["final_choice"] == "tie").sum()
        n = len(df)
        dec = a + b
        rate = a / dec * 100 if dec else 0
        per_chap.append((f.name, a, b, t, n, dec, rate))
        a_tot += a; b_tot += b; t_tot += t; n_tot += n
    dec_tot = a_tot + b_tot
    rate_tot = a_tot / dec_tot * 100 if dec_tot else 0
    p_val = binomtest(min(a_tot, b_tot), dec_tot, 0.5, alternative="less").pvalue if dec_tot else float("nan")

    print(f"\n=== {label} ===")
    print(f"{'file':<70} {'a':>4} {'b':>4} {'tie':>4} {'n':>5} {'dec':>5} {'a%':>6}")
    for row in per_chap:
        if len(row) == 2:
            print(f"{row[0]:<70} {row[1]}")
        else:
            n, a, b, t, total, dec, r = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            print(f"{n:<70} {a:>4} {b:>4} {t:>4} {total:>5} {dec:>5} {r:>5.1f}%")
    print(f"{'TOTAL':<70} {a_tot:>4} {b_tot:>4} {t_tot:>4} {n_tot:>5} {dec_tot:>5} {rate_tot:>5.1f}%")
    print(f"   one-sided sign test p = {p_val:.6f}")
    return {"a": a_tot, "b": b_tot, "tie": t_tot, "n": n_tot, "dec": dec_tot, "rate": rate_tot, "p": p_val}


def main():
    chaps = [6, 7, 8, 9, 10]
    results = {}

    # Within-pipeline ablations
    # chap 6 lives in pairwise_ablation_chap6_... (62 rows, canonical);
    # pairwise_chap6_TMon_vs_TMoff_... never existed.
    results["TM_on vs TM_off"] = agg(
        [JUDGE_DIR / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"]
        + [JUDGE_DIR / f"pairwise_chap{c}_TMon_vs_TMoff_deepseek-chat.csv" for c in chaps[1:]],
        "A3 Translation Memory: TM_on (A) vs TM_off (B), chap 6-10"
    )

    results["Hygiene vs TM_on"] = agg(
        [JUDGE_DIR / f"pairwise_chap{c}_Hygiene_vs_TMon_deepseek-chat.csv" for c in chaps],
        "Hygiene (A) vs TM_on (B), chap 6-10"
    )

    results["A2 vs Hygiene"] = agg(
        [JUDGE_DIR / f"pairwise_chap{c}_A2_vs_Hygiene_deepseek-chat.csv" for c in chaps],
        "A2 (A) vs Hygiene (B), chap 6-10"
    )

    # vs GG
    results["TM_on vs GG"] = agg(
        [JUDGE_DIR / f"pairwise_chap{c}_TMon_vs_GG_deepseek-chat.csv" for c in chaps],
        "TM_on (A) vs GG (B), chap 6-10"
    )
    results["TM_off vs GG (chap 6 only)"] = agg(
        [JUDGE_DIR / "pairwise_chap6_TMoff_vs_GG_deepseek-chat.csv"],
        "TM_off (A) vs GG (B), chap 6 only"
    )
    results["Hygiene vs GG"] = agg(
        [JUDGE_DIR / f"pairwise_chap{c}_Hygiene_vs_GG_deepseek-chat.csv" for c in chaps],
        "Hygiene (A) vs GG (B), chap 6-10"
    )
    results["A2 vs GG"] = agg(
        [JUDGE_DIR / f"pairwise_chap{c}_A2_vs_GG_deepseek-chat.csv" for c in chaps],
        "A2 (A) vs GG (B), chap 6-10"
    )
    results["RAG_BIG (70B) vs GG"] = agg(
        [JUDGE_DIR / f"pairwise_chap{c}_Track3_RAGBIG_vs_GG_deepseek-chat.csv" for c in chaps],
        "Track 3 RAG_BIG (A) vs GG (B), chap 6-10"
    )

    # 70B RAG vs NORAG (chap 7 mini-test)
    results["Track 3 RAG vs NORAG (chap 7)"] = agg(
        [JUDGE_DIR / "pairwise_chap7_Track3_RAGBIG_vs_NORAGBIG_deepseek-chat.csv"],
        "Track 3 RAG_BIG (A) vs NORAG_BIG (B), chap 7 mini-test"
    )
    results["Track 3 NORAG vs GG (chap 7)"] = agg(
        [JUDGE_DIR / "pairwise_chap7_Track3_NORAGBIG_vs_GG_deepseek-chat.csv"],
        "Track 3 NORAG_BIG (A) vs GG (B), chap 7 mini-test"
    )

    # DeepSeek + Claude judge
    df = pd.read_csv("evaluation/llm_judge/claude_judge_deepseek_vs_gg_results.csv", encoding="utf-8-sig")
    print("\n=== DeepSeek-chat (A) vs GG (B) — Claude Opus 4.7 as judge, n=30 stratified ===")
    a = (df["winner"] == "DEEPSEEK").sum()
    b = (df["winner"] == "GG").sum()
    t = (df["winner"] == "TIE").sum()
    dec = a + b
    p = binomtest(min(a, b), dec, 0.5, alternative="less").pvalue
    print(f"DEEPSEEK={a}  GG={b}  TIE={t}  n={len(df)}  dec={dec}  a%={a/dec*100:.1f}%  p={p:.6f}")

    print("\n" + "="*80)
    print("SUMMARY TABLE FOR PAPER SECTION V")
    print("="*80)
    print(f"{'Comparison':<35} {'n_total':>8} {'n_dec':>6} {'A%':>6} {'p':>10}")
    print("-"*80)
    for label, r in results.items():
        print(f"{label:<35} {r['n']:>8} {r['dec']:>6} {r['rate']:>5.1f}% {r['p']:>10.6f}")
    print(f"{'DeepSeek vs GG (Claude judge)':<35} {30:>8} {dec:>6} {a/dec*100:>5.1f}% {p:>10.6f}")


if __name__ == "__main__":
    main()
