#!/usr/bin/env bash
# Track 3 mini-test: Groq llama-3.3-70b-versatile translator on chap 7+8.
# Switched from Gemini (free tier exhausted 20 RPD) to Groq (1000 RPD).
#
# GROQ_SLEEP_SEC=3 disables the overcautious 30s throttle (Groq actual limit
# is 30 RPM = 2s/call minimum, so 3s is conservative).
#
# Two variants:
#   RAG_BIG   = llama3.3-70b + hygiene PostEditing pipeline (glossary + TM + hard_rules)
#   NORAG_BIG = llama3.3-70b + SimplePipeline (no KB, hard_rules-only)
#
# ETA: ~3-5s/call × 90 calls/chap (RAG) + 30 calls/chap (NORAG) = ~10 min/run × 2 chap × 2 variants
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'
export GROQ_SLEEP_SEC=3

echo "=== RAG_BIG: llama3.3-70b + hygiene (chap 7+8) ==="
python main.py \
  --model llama3.3-70b \
  --rag yes \
  --pipeline postedit \
  --tm \
  --experiment-suffix hygiene \
  --chap-start 7 \
  --chap-end 8 \
  --non-interactive 2>&1

echo ""
echo "=== NORAG_BIG: llama3.3-70b + SimplePipeline (chap 7+8) ==="
python main.py \
  --model llama3.3-70b \
  --rag no \
  --chap-start 7 \
  --chap-end 8 \
  --non-interactive 2>&1

echo "DONE"
