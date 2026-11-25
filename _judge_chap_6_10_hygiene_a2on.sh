#!/usr/bin/env bash
# Pairwise judge for A2 ablation: a2_on vs hygiene_on (within-pipeline lift of
# coref-based injection) + a2_on vs GG (does it close any of the residual gap?).
# Two-call position-bias-swap deepseek-chat, same protocol as hygiene run.
#
# Cost: ~$1-2 deepseek-chat for 5 chapters × 2 comparisons.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd "$(dirname "$0")"

CSV_DIR=evaluation/alignment_results_qwen3-8b-local-a2
OUT_DIR=evaluation/llm_judge

# Merge GG column from existing chap-6-10 ablation alignment into the a2 csv.
for n in 6 7 8 9 10; do
  python - <<EOF
import pandas as pd
gg_path = "evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_${n}_with_GG.csv"
a2_path = "$CSV_DIR/aligned_master_chap_${n}_a2.csv"
gg = pd.read_csv(gg_path, encoding="utf-8-sig")
a2 = pd.read_csv(a2_path, encoding="utf-8-sig")
merged = a2.merge(gg[["Source_ID", "GG"]], on="Source_ID", how="left")
out = "$CSV_DIR/aligned_master_chap_${n}_a2_with_GG.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"chap ${n}: a2 n={len(a2)} -> merged n={len(merged)} with GG -> {out}")
EOF
done

run_judge() {
  local chap=$1 col_a=$2 col_b=$3 tag=$4
  echo "=== chap $chap : $col_a vs $col_b ==="
  python batch_pairwise_judge.py \
    --review "$CSV_DIR/aligned_master_chap_${chap}_a2_with_GG.csv" \
    --col-a "$col_a" --col-b "$col_b" \
    --col-source Source_Text --col-ref ref --col-id Source_ID \
    --default-chapter "$chap" \
    --judge deepseek-chat \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --out "$OUT_DIR/pairwise_chap${chap}_${tag}_deepseek-chat.csv" \
    --sleep 1.0 2>&1 | tail -3
}

# A. a2_on vs hygiene_on (marginal lift of A2 on top of hygiene)
for n in 6 7 8 9 10; do
  run_judge $n a2_on hygiene_on A2_vs_Hygiene
done

# B. a2_on vs GG (residual gap after A2)
for n in 6 7 8 9 10; do
  run_judge $n a2_on GG A2_vs_GG
done

echo "ALL A2 JUDGES DONE"
