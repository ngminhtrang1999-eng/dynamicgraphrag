#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd "$(dirname "$0")"
python -u main.py \
  --model qwen3-8b-local \
  --rag yes \
  --pipeline postedit \
  --no-tm \
  --experiment-suffix notm \
  --chap-start 7 \
  --chap-end 10 \
  --non-interactive 2>&1
echo "DONE"
