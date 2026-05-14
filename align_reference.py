"""Align an EN source chapter against a curated VN reference (no RAG candidates).

Used to expand the Translation Memory seed pool with judge-validated user-edited
references for chapters that lack model translations (e.g. chap 11-19 edited
through vntq_edit_app).

Output: aligned_master_chap_<N>.csv with columns Source_ID, Source_Text, ref.
The schema matches what seed_translation_memory.py expects.

Usage:
    python align_reference.py \\
        --chap 11 \\
        --src data/LOTM/chapters/chapter_0011.txt \\
        --ref data/LOTM/chapters_vi/chap_11.txt \\
        --out-dir evaluation/alignment_results_reference

    # Batch chap 11-19:
    for n in 11 12 13 14 15 16 17 18 19; do
        python align_reference.py --chap $n \\
            --src "data/LOTM/chapters/chapter_$(printf %04d $n).txt" \\
            --ref "data/LOTM/chapters_vi/chap_${n}.txt" \\
            --out-dir evaluation/alignment_results_reference
    done
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from align_ablation import (
    MODEL_NAME,
    fill_column,
    global_dynamic_alignment,
    smart_sentence_split,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chap", type=int, required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    for p in [args.src, args.ref]:
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME).to(device)

    def read(p):
        with open(p, "r", encoding="utf-8-sig") as f:
            return f.read()

    src_sents = smart_sentence_split(read(args.src))
    ref_sents = smart_sentence_split(read(args.ref))
    print(f"  source: {len(src_sents)} sents | ref: {len(ref_sents)}")

    print("Aligning...")
    pairs = global_dynamic_alignment(src_sents, ref_sents, model, device)

    df = pd.DataFrame(
        {
            "Source_ID": list(range(len(src_sents))),
            "Source_Text": src_sents,
            "ref": fill_column(src_sents, pairs),
        }
    )
    out_path = os.path.join(args.out_dir, f"aligned_master_chap_{args.chap}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
