#!/usr/bin/env bash
# Gemini re-judge of the n=500 stratified sample from the chapters-6-10 evidence.
set -uo pipefail
cd "/mnt/d/Dynamic GraphRAG"
export PYTHONIOENCODING=utf-8

GEM_URL=https://generativelanguage.googleapis.com/v1beta/openai/
JUDGE=gemini-3.5-flash

for NAME in TMon_vs_TMoff Hygiene_vs_TMon A2_vs_Hygiene A2_vs_GG RAGBIG_vs_GG; do
  python batch_pairwise_judge.py \
    --review "evaluation/llm_judge/rejudge_${NAME}.csv" \
    --judge "$JUDGE" --base-url "$GEM_URL" --api-key-env GEMINI_API_KEY \
    --sleep 0.6 \
    --out "evaluation/llm_judge/rejudge_${NAME}_gemini35flash.csv" \
    && echo "[ok] rejudge $NAME" || echo "[FAIL] rejudge $NAME"
done
echo REJUDGE_DONE
