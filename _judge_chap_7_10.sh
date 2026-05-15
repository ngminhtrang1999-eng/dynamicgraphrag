#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'

CSV_DIR=evaluation/alignment_results_qwen3-8b-local-ablation
OUT_DIR=evaluation/llm_judge

run_judge() {
  local chap=$1 col_a=$2 col_b=$3 tag=$4
  echo "=== chap $chap : $col_a vs $col_b ==="
  python batch_pairwise_judge.py \
    --review "$CSV_DIR/aligned_master_chap_${chap}_with_GG.csv" \
    --col-a "$col_a" --col-b "$col_b" \
    --col-source Source_Text --col-ref ref --col-id Source_ID \
    --default-chapter "$chap" \
    --judge deepseek-chat \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --out "$OUT_DIR/pairwise_chap${chap}_${tag}_deepseek-chat.csv" \
    --sleep 1.0 2>&1 | tail -3
}

# A. TM_on vs TM_off for chap 7-10 (ablation main claim)
for n in 7 8 9 10; do
  run_judge $n TM_on TM_off TMon_vs_TMoff
done

# B. TM_on vs GG for chap 7-10 (thesis: RAG > GG?)
for n in 7 8 9 10; do
  run_judge $n TM_on GG TMon_vs_GG
done

echo "ALL JUDGES DONE"
