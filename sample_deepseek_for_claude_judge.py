"""Generate stratified sample (6 pairs/chap × 5 chap = 30 pairs) of
DeepSeek-chat vs GG for Claude (in-conversation) pairwise judging.

Per-pair random A/B assignment removes position bias from the single-call
judge protocol. Output CSV columns:
    pair_id, chap, sid, src, ref, A_text, B_text, A_system, B_system

Reproducible via --seed.
"""
import argparse
import random
import pandas as pd
from pathlib import Path

ALIGNED_DIR = Path("evaluation/alignment_results_deepseek_translator")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chap", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument("--out", default="evaluation/llm_judge/claude_sample_deepseek_vs_gg.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    pair_id = 0

    for chap in [6, 7, 8, 9, 10]:
        path = ALIGNED_DIR / f"aligned_master_chap_{chap}_deepseek.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        valid = df[
            df["DEEPSEEK"].notna() & df["GG"].notna() & df["ref"].notna()
            & (df["DEEPSEEK"] != "<MERGED_ABOVE>")
            & (df["GG"] != "<MERGED_ABOVE>")
            & (df["ref"] != "<MERGED_ABOVE>")
            & df["DEEPSEEK"].astype(str).str.strip().ne("")
            & df["GG"].astype(str).str.strip().ne("")
            & df["ref"].astype(str).str.strip().ne("")
        ].copy()

        indices = rng.sample(list(valid.index), min(args.per_chap, len(valid)))
        for idx in sorted(indices):
            r = valid.loc[idx]
            # Random A/B assignment
            if rng.random() < 0.5:
                a_sys, a_text = "DEEPSEEK", r["DEEPSEEK"]
                b_sys, b_text = "GG", r["GG"]
            else:
                a_sys, a_text = "GG", r["GG"]
                b_sys, b_text = "DEEPSEEK", r["DEEPSEEK"]
            pair_id += 1
            rows.append({
                "pair_id": pair_id,
                "chap": chap,
                "sid": int(r["Source_ID"]),
                "src": r["Source_Text"],
                "ref": r["ref"],
                "A_text": a_text,
                "B_text": b_text,
                "A_system": a_sys,
                "B_system": b_sys,
            })

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out_df)} pairs to {args.out}")
    print(f"A_system distribution: {out_df['A_system'].value_counts().to_dict()}")
    print(f"Per chap: {out_df['chap'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
