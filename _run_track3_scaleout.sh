#!/usr/bin/env bash
# Track 3 scale-out: Groq llama-3.3-70b-versatile RAG_BIG (hygiene PostEdit)
# Extends mini-test (chap 7 only) to chap 6, 8, 9, 10. Chap 7 already done -
# pipeline will skip it because translation_progress.json keeps its entry.
#
# Run mode per user decision (2026-05-16):
#   - Sequential 6->8->9->10 (chap 7 preserved from mini-test).
#   - Single run with resume-on-fail (pipeline tracks progress per chap).
#   - Groq throttle GROQ_SLEEP_SEC=3 (mini-test value, ~30 RPM safe).
#   - If Groq TPD hit, script fails partway; rerun resumes.
#
# Skip NORAG_BIG: chap 7 mini-test established RAG > NORAG at 70B
# (69.3% decisive, p=0.0005). Scale-out only needs RAG_BIG vs GG.
#
# ETA: ~6-10 min wall-clock per chap, ~30-40 min total for 4 chaps.

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd "$(dirname "$0")"
export GROQ_SLEEP_SEC=3

LOG=track3_scaleout_$(date +%Y%m%d_%H%M%S).log
echo "=== Track 3 scale-out: llama-3.3-70b RAG_BIG chap 6, 8, 9, 10 ===" | tee -a "$LOG"
echo "Start: $(date -Iseconds)" | tee -a "$LOG"
echo "KB pre-run: TM=$(python3 -c 'import json;print(len(json.load(open("data/knowledge_base/llama3.3-70b_yes_hygiene/translation_memory.json"))))') glossary=$(python3 -c 'import json;print(len(json.load(open("data/knowledge_base/llama3.3-70b_yes_hygiene/term_glossary.json"))))')" | tee -a "$LOG"

python main.py \
  --model llama3.3-70b \
  --rag yes \
  --pipeline postedit \
  --tm \
  --experiment-suffix hygiene \
  --chap-start 6 \
  --chap-end 10 \
  --non-interactive 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "End: $(date -Iseconds)" | tee -a "$LOG"
echo "KB post-run: TM=$(python3 -c 'import json;print(len(json.load(open("data/knowledge_base/llama3.3-70b_yes_hygiene/translation_memory.json"))))') glossary=$(python3 -c 'import json;print(len(json.load(open("data/knowledge_base/llama3.3-70b_yes_hygiene/term_glossary.json"))))')" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Output files:" | tee -a "$LOG"
ls -la data/LOTM/translations/trans_chap_{6,7,8,9,10}_llama3.3-70b_yes_hygiene.txt 2>&1 | tee -a "$LOG"

echo "DONE" | tee -a "$LOG"
