"""Unified COMET + BLEURT + chrF evaluator.

Reads aligned CSVs, scores each candidate column against the ref column, writes
per-row scores + a summary table. Designed to be re-runnable as new alignment
files arrive (e.g. Track 3 chap 9+10 after Groq resume).

Backbone for objective metrics that don't depend on the LLM judge, so we can
still compare deepseek-chat or other near-judge-capability translators.

Usage:
    python batch_evaluate_metrics.py \\
        --inputs "evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_{6..10}_with_GG.csv" \\
        --columns TM_on TM_off GG \\
        --metrics comet chrf \\
        --out-dir evaluation/objective_metrics/qwen3-8b-ablation

    python batch_evaluate_metrics.py \\
        --inputs "evaluation/alignment_results_track3/aligned_master_chap_*_track3.csv" \\
        --columns RAG_BIG GG \\
        --metrics comet bleurt chrf \\
        --out-dir evaluation/objective_metrics/track3

Notes:
- BLEURT checkpoint located at ./BLEURT-20 (override with --bleurt-checkpoint).
- COMET uses Unbabel/wmt22-comet-da (auto-downloads to HF cache).
- chrF via sacrebleu (chrF++ variant, no extra setup).
- Rows where the candidate is empty or "<MERGED_ABOVE>" are dropped per metric.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


def _clean_pairs(df: pd.DataFrame, src_col: str, ref_col: str, cand_col: str):
    """Return (src, mt, ref) triplets dropping empty / merged-above / missing rows."""
    rows = []
    for _, r in df.iterrows():
        ref = r.get(ref_col)
        cand = r.get(cand_col)
        src = r.get(src_col, "")
        if not isinstance(ref, str) or not isinstance(cand, str):
            continue
        if not ref.strip() or not cand.strip():
            continue
        if cand.strip() == "<MERGED_ABOVE>" or ref.strip() == "<MERGED_ABOVE>":
            continue
        rows.append({"src": str(src), "mt": cand, "ref": ref})
    return rows


def score_comet(triplets, gpus: int):
    from comet import download_model, load_from_checkpoint
    model_id = "Unbabel/wmt22-comet-da"
    print(f"  [COMET] loading {model_id}")
    model = load_from_checkpoint(download_model(model_id))
    out = model.predict(triplets, batch_size=8, gpus=gpus, progress_bar=False)
    return list(out["scores"]), float(out["system_score"])


def score_bleurt(triplets, checkpoint: str):
    from bleurt import score as bleurt_score
    print(f"  [BLEURT] loading {checkpoint}")
    scorer = bleurt_score.BleurtScorer(checkpoint)
    refs = [t["ref"] for t in triplets]
    cands = [t["mt"] for t in triplets]
    scores = scorer.score(references=refs, candidates=cands, batch_size=16)
    return [float(s) for s in scores], float(sum(scores) / max(1, len(scores)))


def score_chrf(triplets):
    import sacrebleu
    refs = [t["ref"] for t in triplets]
    cands = [t["mt"] for t in triplets]
    per_row = [
        sacrebleu.sentence_chrf(c, [r], word_order=2).score
        for c, r in zip(cands, refs)
    ]
    sys_score = sacrebleu.corpus_chrf(cands, [refs], word_order=2).score
    return per_row, float(sys_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Glob pattern for aligned CSVs")
    ap.add_argument("--columns", nargs="+", required=True, help="Candidate columns to score")
    ap.add_argument("--src-col", default="Source_Text")
    ap.add_argument("--ref-col", default="ref")
    ap.add_argument("--metrics", nargs="+", default=["comet", "chrf"],
                    choices=["comet", "bleurt", "chrf"])
    ap.add_argument("--bleurt-checkpoint", default="./BLEURT-20")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.inputs))
    if not files:
        sys.exit(f"No CSVs matched: {args.inputs}")
    os.makedirs(args.out_dir, exist_ok=True)
    gpus = 1 if torch.cuda.is_available() else 0
    print(f"Found {len(files)} input file(s). GPUs={gpus}. Metrics={args.metrics}.")

    summary = []
    for fpath in files:
        df = pd.read_csv(fpath, encoding="utf-8-sig")
        chap_tag = Path(fpath).stem
        print(f"\n=== {chap_tag} (n={len(df)}) ===")
        for col in args.columns:
            if col not in df.columns:
                print(f"  SKIP {col}: column missing")
                continue
            triplets = _clean_pairs(df, args.src_col, args.ref_col, col)
            if not triplets:
                print(f"  SKIP {col}: no valid rows")
                continue
            print(f"  {col}: n_clean={len(triplets)}")
            row_records = [{"src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triplets]

            for metric in args.metrics:
                if metric == "comet":
                    per_row, sys_score = score_comet(triplets, gpus)
                elif metric == "bleurt":
                    per_row, sys_score = score_bleurt(triplets, args.bleurt_checkpoint)
                elif metric == "chrf":
                    per_row, sys_score = score_chrf(triplets)
                else:
                    continue
                for rec, s in zip(row_records, per_row):
                    rec[metric] = s
                summary.append({
                    "file": chap_tag, "system": col, "metric": metric,
                    "n": len(triplets), "score": sys_score,
                })
                print(f"    {metric:7s} = {sys_score:.4f}")

            out_csv = os.path.join(args.out_dir, f"{chap_tag}__{col}.csv")
            pd.DataFrame(row_records).to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.out_dir, "_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSummary written to {summary_path}")
    print(summary_df.to_string(index=False))

    # Aggregate by (system, metric) across all files
    if not summary_df.empty:
        agg = summary_df.groupby(["system", "metric"]).agg(
            n_total=("n", "sum"),
            score_mean=("score", "mean"),
        ).reset_index()
        agg_path = os.path.join(args.out_dir, "_aggregate.csv")
        agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
        print(f"\nAggregate written to {agg_path}")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
