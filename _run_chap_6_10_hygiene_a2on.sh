#!/usr/bin/env bash
# A2 ablation generation — fastcoref-based deterministic entity injection +
# tight (coref-anchored) glossary filter, layered on top of the hygiene baseline.
#
# Why hygiene baseline: hygiene is the current strongest in-pipeline pipeline
# (55.0% vs TM_on, p=0.053). Track 1 qualitative analysis predicts A2 should
# add ~1-3pp on register axis — small effect, needs the strongest baseline
# to be a clean test. See evaluation/analysis/qualitative_gap_analysis.md
# section 4 "Design implications for A2".
#
# Pre-step: clone the hygiene data dir so A2 has its own translation_progress
# tracking (and we don't overwrite the existing hygiene outputs). The TM,
# glossary, and graph state are shared, so the comparison isolates the
# injection/glossary-filter change exclusively.
#
# Estimated runtime: ~75 min/chap × 5 ≈ 6h on qwen3:8b Ollama, fastcoref on
# GPU adds ~100ms per chunk (negligible).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'

SRC=data/knowledge_base/qwen3-8b-local_yes_hygiene
DST=data/knowledge_base/qwen3-8b-local_yes_hygiene_a2on

if [ ! -d "$DST" ]; then
  echo "Cloning $SRC -> $DST so A2 has its own progress tracking..."
  cp -r "$SRC" "$DST"
  # Drop the translation-progress file so we re-run all chunks under A2.
  rm -f "$DST/translation_progress.json"
fi

python main.py \
  --model qwen3-8b-local \
  --rag yes \
  --pipeline postedit \
  --tm \
  --a2-coref \
  --experiment-suffix hygiene_a2on \
  --chap-start 6 \
  --chap-end 10 \
  --non-interactive 2>&1
echo "DONE"
