#!/usr/bin/env bash
# Align + judge frontier-Chinese-rich runs (qwen3-235b, DeepSeek-V3.1).
#
# qwen3-235b: judge deepseek-chat OK (different family, borderline)
#   → pairwise judge + chrF + COMET.
#
# DeepSeek-V3.1: SAME FAMILY as judge → self-bias risk
#   → SKIP pairwise judge entirely. Only chrF + COMET.
#
# Usage: bash _align_judge_frontier.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'

LOG="frontier_align_judge_$(date +%Y%m%d_%H%M%S).log"
OUT_DIR=evaluation/alignment_results_frontier
mkdir -p "$OUT_DIR"
TRANS_DIR=data/LOTM/translations

echo "=== Align + judge frontier-Chinese-rich runs ===" | tee -a "$LOG"
echo "Start: $(date -Iseconds)" | tee -a "$LOG"

align_one_model() {
  local model_tag=$1   # qwen3-235b-a22b  OR  deepseek-v3.1
  local col_name=$2    # QWEN235B  OR  DEEPSEEK_V31

  for chap in 6 7 8 9 10; do
    local src_file="data/LOTM/chapters/chapter_$(printf '%04d' $chap).txt"
    local ref_file="data/LOTM/translations/reference/ref_${chap}.txt"
    local cand_file="$TRANS_DIR/trans_chap_${chap}_${model_tag}_yes_hygiene.txt"
    local gg_src="evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_${chap}_with_GG.csv"

    if [ ! -f "$cand_file" ]; then
      echo "  SKIP chap $chap: $cand_file missing" | tee -a "$LOG"
      continue
    fi

    echo "=== Aligning chap $chap ($model_tag) ===" | tee -a "$LOG"
    python align_ablation.py --chap "$chap" --src "$src_file" --ref "$ref_file" \
      --tm-on "$cand_file" --tm-off "$cand_file" \
      --out-dir "$OUT_DIR" 2>&1 | tail -3 | tee -a "$LOG"

    python - <<EOF | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("$OUT_DIR/aligned_master_chap_${chap}.csv", encoding="utf-8-sig")
df = df.drop(columns=["TM_off"]).rename(columns={"TM_on": "$col_name"})
gg = pd.read_csv("$gg_src", encoding="utf-8-sig")
merged = df.merge(gg[["Source_ID","GG"]], on="Source_ID", how="left")
out = "$OUT_DIR/aligned_master_chap_${chap}_${model_tag}.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"chap ${chap} ${model_tag}: n={len(merged)} -> {out}")
EOF
  done
}

judge_vs_gg() {
  local model_tag=$1
  local col_name=$2

  for chap in 6 7 8 9 10; do
    local aligned="$OUT_DIR/aligned_master_chap_${chap}_${model_tag}.csv"
    if [ ! -f "$aligned" ]; then continue; fi

    echo "=== chap $chap : $col_name vs GG (deepseek-chat) ===" | tee -a "$LOG"
    python batch_pairwise_judge.py \
      --review "$aligned" \
      --col-a "$col_name" --col-b GG \
      --col-source Source_Text --col-ref ref --col-id Source_ID \
      --default-chapter "$chap" \
      --judge deepseek-chat \
      --base-url https://api.deepseek.com \
      --api-key-env DEEPSEEK_API_KEY \
      --out "evaluation/llm_judge/pairwise_chap${chap}_${col_name}_vs_GG_deepseek-chat.csv" \
      --sleep 1.0 2>&1 | tail -3 | tee -a "$LOG"
  done
}

echo "" | tee -a "$LOG"
echo "[1/4] Aligning qwen3-235b-a22b..." | tee -a "$LOG"
align_one_model qwen3-235b-a22b QWEN235B

echo "" | tee -a "$LOG"
echo "[2/4] Aligning deepseek-v3.1..." | tee -a "$LOG"
align_one_model deepseek-v3.1 DEEPSEEK_V31

echo "" | tee -a "$LOG"
echo "[3/4] Pairwise judge: qwen3-235b vs GG (deepseek-chat judge OK)..." | tee -a "$LOG"
judge_vs_gg qwen3-235b-a22b QWEN235B

echo "" | tee -a "$LOG"
echo "[4/4] DeepSeek-V3.1: SKIP pairwise judge (same family as judge)" | tee -a "$LOG"
echo "      Objective metrics only — chrF + COMET via batch_evaluate_metrics.py" | tee -a "$LOG"

# Compute objective metrics for both
echo "" | tee -a "$LOG"
echo "[Metrics] chrF + COMET for both frontier models..." | tee -a "$LOG"
python batch_evaluate_metrics.py \
  --inputs "$OUT_DIR/aligned_master_chap_*_qwen3-235b-a22b.csv" \
  --columns QWEN235B GG \
  --metrics chrf comet \
  --out-dir evaluation/objective_metrics/frontier_qwen235b 2>&1 | tail -15 | tee -a "$LOG"

python batch_evaluate_metrics.py \
  --inputs "$OUT_DIR/aligned_master_chap_*_deepseek-v3.1.csv" \
  --columns DEEPSEEK_V31 GG \
  --metrics chrf comet \
  --out-dir evaluation/objective_metrics/frontier_deepseek_v31 2>&1 | tail -15 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "End: $(date -Iseconds)" | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"
