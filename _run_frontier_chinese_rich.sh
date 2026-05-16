#!/usr/bin/env bash
# Frontier-Chinese-rich scale-out: tests "domain fit > parameter count" directly.
# Run AFTER Track 3 chap 9, 10 lands (i.e., after _resume_track3_chap9_10.sh).
#
# Two open-frontier Chinese-rich translators:
#   1. qwen3-235b-a22b — OpenRouter free, 235B MoE (22B active), Chinese-rich
#      training, same lineage as qwen3:8b base. judge: deepseek-chat OK.
#   2. DeepSeek-V3.1-terminus — NVIDIA NIM free, 671B MoE, Chinese-rich.
#      JUDGE CONCERN: deepseek family → self-bias risk → COMET/chrF only.
#
# Skips gemma3-27b (English-trained mid-tier; Track 3 70B llama already
# established that English-rich scaling doesn't help; gemma3-27b would
# replicate that finding without adding new info).
#
# Each run: ~5 chap × ~6-8 chunks × ~3-5 LLM calls per chunk = ~120-200 calls.
# OpenRouter free has per-model daily limits; may need throttling.
#
# Usage: bash _run_frontier_chinese_rich.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'

LOG="frontier_chinese_$(date +%Y%m%d_%H%M%S).log"

echo "=== Frontier-Chinese-rich scale-out ===" | tee -a "$LOG"
echo "Start: $(date -Iseconds)" | tee -a "$LOG"

# Reset progress for clean run — these are FRESH experiment dirs
# (different model names → main.py picks different experiment_sub_dir).

echo "" | tee -a "$LOG"
echo "[1/2] qwen3-235b-a22b (OpenRouter free)" | tee -a "$LOG"
python main.py \
  --model qwen3-235b-a22b \
  --rag yes \
  --pipeline postedit \
  --tm \
  --experiment-suffix hygiene \
  --chap-start 6 \
  --chap-end 10 \
  --non-interactive 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[2/2] DeepSeek-V3.1-terminus (NVIDIA NIM free)" | tee -a "$LOG"
python main.py \
  --model deepseek-v3.1 \
  --rag yes \
  --pipeline postedit \
  --tm \
  --experiment-suffix hygiene \
  --chap-start 6 \
  --chap-end 10 \
  --non-interactive 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "End: $(date -Iseconds)" | tee -a "$LOG"
echo "Output files:" | tee -a "$LOG"
ls -la data/LOTM/translations/trans_chap_{6,7,8,9,10}_qwen3-235b-a22b_yes_hygiene.txt 2>&1 | tee -a "$LOG"
ls -la data/LOTM/translations/trans_chap_{6,7,8,9,10}_deepseek-v3.1_yes_hygiene.txt 2>&1 | tee -a "$LOG"

echo "DONE - next step: bash _align_judge_frontier.sh"
