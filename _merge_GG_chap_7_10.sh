#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'
python - <<'PY'
import pandas as pd
from pathlib import Path

ABLATION_DIR = Path("evaluation/alignment_results_qwen3-8b-local-ablation")
LLAMA_DIR = Path("evaluation/alignment_results_llama-3.1-8B")

for n in [7, 8, 9, 10]:
    a_path = ABLATION_DIR / f"aligned_master_chap_{n}.csv"
    b_path = LLAMA_DIR / f"aligned_master_chap_{n}.csv"
    if not a_path.exists():
        print(f"skip {n}: {a_path} missing")
        continue
    if not b_path.exists():
        print(f"skip {n}: {b_path} missing")
        continue
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    merged = a.merge(b[["Source_ID", "GG"]], on="Source_ID", how="left")
    n_match = merged["GG"].notna().sum()
    out = ABLATION_DIR / f"aligned_master_chap_{n}_with_GG.csv"
    merged.to_csv(out, index=False)
    print(f"chap {n}: {len(merged)} rows, GG non-null={n_match}, saved {out}")
PY
echo "DONE"
