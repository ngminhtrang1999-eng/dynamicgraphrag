#!/usr/bin/env bash
# MANUAL RESUME script for Track 3 scale-out chap 9 + 10.
# Run anytime after Groq TPD reset (typically midnight UTC = 07:00 VN time).
#
# What this does:
#   1. Quick Groq quota probe (5-token call).
#   2. Resume generation chap 9 + 10 (chap 6, 7, 8 marked done in progress.json, skipped).
#   3. Align + judge chap 9, 10 against GG.
#
# Usage: bash _resume_track3_chap9_10.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd "$(dirname "$0")"

LOG="track3_resume_$(date +%Y%m%d_%H%M%S).log"
echo "=== Track 3 RESUME chap 9, 10 ===" | tee -a "$LOG"
echo "Start: $(date -Iseconds)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[1/3] Groq quota probe..." | tee -a "$LOG"
python3 -c "
from groq import Groq
import os
c = Groq(api_key=os.getenv('GROQ_API_KEY'))
r = c.chat.completions.create(model='llama-3.3-70b-versatile', messages=[{'role':'user','content':'ok'}], max_tokens=5)
print('Probe OK:', r.choices[0].message.content[:20])
" 2>&1 | tee -a "$LOG" || { echo "GROQ QUOTA NOT READY - try again later" | tee -a "$LOG"; exit 1; }

echo "" | tee -a "$LOG"
echo "[2/3] Generation chap 9 + 10 (resume via progress.json)..." | tee -a "$LOG"
export GROQ_SLEEP_SEC=3
python main.py \
  --model llama3.3-70b \
  --rag yes \
  --pipeline postedit \
  --tm \
  --experiment-suffix hygiene \
  --chap-start 9 \
  --chap-end 10 \
  --non-interactive 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[3/3] Align + judge chap 9, 10..." | tee -a "$LOG"

OUT_DIR=evaluation/alignment_results_track3
TRANS_DIR=data/LOTM/translations

align_and_merge_gg() {
  local chap=$1
  local src_file="data/LOTM/chapters/chapter_$(printf '%04d' $chap).txt"
  local ref_file="data/LOTM/translations/reference/ref_${chap}.txt"
  local rag_big="$TRANS_DIR/trans_chap_${chap}_llama3.3-70b_yes_hygiene.txt"
  local gg_src="evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_${chap}_with_GG.csv"

  python align_ablation.py --chap "$chap" --src "$src_file" --ref "$ref_file" \
    --tm-on "$rag_big" --tm-off "$rag_big" --out-dir "$OUT_DIR" 2>&1 | tail -3

  python - <<EOF
import pandas as pd
df = pd.read_csv("$OUT_DIR/aligned_master_chap_${chap}.csv", encoding="utf-8-sig")
df = df.drop(columns=["TM_off"]).rename(columns={"TM_on": "RAG_BIG"})
gg = pd.read_csv("$gg_src", encoding="utf-8-sig")
merged = df.merge(gg[["Source_ID","GG"]], on="Source_ID", how="left")
merged.to_csv("$OUT_DIR/aligned_master_chap_${chap}_track3.csv", index=False, encoding="utf-8-sig")
print(f"chap ${chap} aligned: n={len(merged)}")
EOF
}

run_judge_vs_gg() {
  local chap=$1
  python batch_pairwise_judge.py \
    --review "$OUT_DIR/aligned_master_chap_${chap}_track3.csv" \
    --col-a RAG_BIG --col-b GG \
    --col-source Source_Text --col-ref ref --col-id Source_ID \
    --default-chapter "$chap" \
    --judge deepseek-chat --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --out "evaluation/llm_judge/pairwise_chap${chap}_Track3_RAGBIG_vs_GG_deepseek-chat.csv" \
    --sleep 1.0 2>&1 | tail -3
}

for chap in 9 10; do
  align_and_merge_gg $chap | tee -a "$LOG"
  run_judge_vs_gg $chap | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "End: $(date -Iseconds)" | tee -a "$LOG"
echo "DONE - Output CSVs:" | tee -a "$LOG"
ls -la evaluation/llm_judge/pairwise_chap{6,7,8,9,10}_Track3_RAGBIG_vs_GG_deepseek-chat.csv 2>&1 | tee -a "$LOG"
