#!/usr/bin/env bash
# Align + objective metrics (chrF + COMET) for deepseek-chat as translator.
# SKIP pairwise judge via deepseek-chat (self-bias).
# Stratified sample 30 pairs will be presented to Claude in conversation.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'

OUT_DIR=evaluation/alignment_results_deepseek_translator
mkdir -p "$OUT_DIR"
TRANS_DIR=data/LOTM/translations

LOG="deepseek_translator_align_$(date +%Y%m%d_%H%M%S).log"
echo "=== Align + objective metrics: deepseek-chat translator chap 6-10 ===" | tee -a "$LOG"

for chap in 6 7 8 9 10; do
  src_file="data/LOTM/chapters/chapter_$(printf '%04d' $chap).txt"
  ref_file="data/LOTM/translations/reference/ref_${chap}.txt"
  cand_file="$TRANS_DIR/trans_chap_${chap}_deepseek-chat_yes_hygiene.txt"
  gg_src="evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_${chap}_with_GG.csv"

  if [ ! -f "$cand_file" ]; then
    echo "  SKIP chap $chap: $cand_file missing" | tee -a "$LOG"
    continue
  fi

  echo "=== Aligning chap $chap ===" | tee -a "$LOG"
  python align_ablation.py --chap "$chap" --src "$src_file" --ref "$ref_file" \
    --tm-on "$cand_file" --tm-off "$cand_file" \
    --out-dir "$OUT_DIR" 2>&1 | tail -3 | tee -a "$LOG"

  python - <<EOF | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("$OUT_DIR/aligned_master_chap_${chap}.csv", encoding="utf-8-sig")
df = df.drop(columns=["TM_off"]).rename(columns={"TM_on": "DEEPSEEK"})
gg = pd.read_csv("$gg_src", encoding="utf-8-sig")
merged = df.merge(gg[["Source_ID","GG"]], on="Source_ID", how="left")
out = "$OUT_DIR/aligned_master_chap_${chap}_deepseek.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"chap ${chap}: n={len(merged)} -> {out}")
EOF
done

echo "" | tee -a "$LOG"
echo "=== chrF + COMET for deepseek-chat vs GG ===" | tee -a "$LOG"
python batch_evaluate_metrics.py \
  --inputs "$OUT_DIR/aligned_master_chap_*_deepseek.csv" \
  --columns DEEPSEEK GG \
  --metrics chrf comet \
  --out-dir evaluation/objective_metrics/deepseek_translator 2>&1 | tail -25 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"
