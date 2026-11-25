#!/usr/bin/env bash
# Pairwise judge: hygiene_on vs TM_on (within-pipeline ablation) + hygiene_on vs GG (vs commercial MT).
# Uses deepseek-chat with the same two-call position-bias-swap protocol.
#
# Estimated cost: ~$1-2 deepseek-chat for 5 chapters × 2 comparisons.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd "$(dirname "$0")"

CSV_DIR=evaluation/alignment_results_qwen3-8b-local-hygiene
OUT_DIR=evaluation/llm_judge

# Merge GG column from the existing chap-6-10 alignment into hygiene csv.
# Reuses _merge_GG_chap_7_10.sh approach: outer-join on Source_ID.
for n in 6 7 8 9 10; do
  python - <<EOF
import pandas as pd
gg_path = "evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_${n}_with_GG.csv"
hy_path = "$CSV_DIR/aligned_master_chap_${n}_hygiene.csv"
gg = pd.read_csv(gg_path, encoding="utf-8-sig")
hy = pd.read_csv(hy_path, encoding="utf-8-sig")
gg_col = gg[["Source_ID", "GG"]]
merged = hy.merge(gg_col, on="Source_ID", how="left")
out = "$CSV_DIR/aligned_master_chap_${n}_hygiene_with_GG.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"chap ${n}: hygiene n={len(hy)} -> merged n={len(merged)} with GG -> {out}")
EOF
done

run_judge() {
  local chap=$1 col_a=$2 col_b=$3 tag=$4
  echo "=== chap $chap : $col_a vs $col_b ==="
  python batch_pairwise_judge.py \
    --review "$CSV_DIR/aligned_master_chap_${chap}_hygiene_with_GG.csv" \
    --col-a "$col_a" --col-b "$col_b" \
    --col-source Source_Text --col-ref ref --col-id Source_ID \
    --default-chapter "$chap" \
    --judge deepseek-chat \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --out "$OUT_DIR/pairwise_chap${chap}_${tag}_deepseek-chat.csv" \
    --sleep 1.0 2>&1 | tail -3
}

# A. hygiene_on vs TM_on baseline (does hygiene help?)
for n in 6 7 8 9 10; do
  run_judge $n hygiene_on TM_on Hygiene_vs_TMon
done

# B. hygiene_on vs GG (did hygiene close the gap to commercial MT?)
for n in 6 7 8 9 10; do
  run_judge $n hygiene_on GG Hygiene_vs_GG
done

echo "ALL JUDGES DONE"
