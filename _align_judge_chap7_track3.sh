#!/usr/bin/env bash
# Track 3 mini-test, chap 7 only (chap 8 NORAG hit Groq TPD limit, dropped per user decision).
# Aligns RAG_BIG + NORAG_BIG against src/ref, merges GG, then runs 3 pairwise judges.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'

OUT_DIR=evaluation/alignment_results_track3
mkdir -p "$OUT_DIR"
TRANS_DIR=data/LOTM/translations

echo "=== Aligning chap 7 (RAG_BIG vs NORAG_BIG against src+ref) ==="
python align_ablation.py \
  --chap 7 \
  --src "data/LOTM/chapters/chapter_0007.txt" \
  --ref "data/LOTM/translations/reference/ref_7.txt" \
  --tm-on  "$TRANS_DIR/trans_chap_7_llama3.3-70b_yes_hygiene.txt" \
  --tm-off "$TRANS_DIR/trans_chap_7_llama3.3-70b_no.txt" \
  --out-dir "$OUT_DIR" 2>&1 | tail -4

# Rename TM_on/TM_off columns → RAG_BIG / NORAG_BIG and merge GG from existing alignment
python - <<EOF
import pandas as pd
p = "$OUT_DIR/aligned_master_chap_7.csv"
df = pd.read_csv(p, encoding="utf-8-sig")
df = df.rename(columns={"TM_on": "RAG_BIG", "TM_off": "NORAG_BIG"})

# Merge GG column from existing aligned_master_chap_7_with_GG.csv on Source_ID
gg = pd.read_csv("evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_7_with_GG.csv", encoding="utf-8-sig")
merged = df.merge(gg[["Source_ID", "GG"]], on="Source_ID", how="left")
out = "$OUT_DIR/aligned_master_chap_7_track3.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"chap 7 track3: n={len(merged)} cols={list(merged.columns)} -> {out}")
EOF

run_judge() {
  local col_a=$1 col_b=$2 tag=$3
  echo "=== chap 7 : $col_a vs $col_b ==="
  python batch_pairwise_judge.py \
    --review "$OUT_DIR/aligned_master_chap_7_track3.csv" \
    --col-a "$col_a" --col-b "$col_b" \
    --col-source Source_Text --col-ref ref --col-id Source_ID \
    --default-chapter 7 \
    --judge deepseek-chat \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --out "evaluation/llm_judge/pairwise_chap7_${tag}_deepseek-chat.csv" \
    --sleep 1.0 2>&1 | tail -3
}

run_judge RAG_BIG NORAG_BIG Track3_RAGBIG_vs_NORAGBIG
run_judge RAG_BIG GG Track3_RAGBIG_vs_GG
run_judge NORAG_BIG GG Track3_NORAGBIG_vs_GG

echo "ALL TRACK3 JUDGES DONE"
