#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd "$(dirname "$0")"
python main.py \
  --model qwen3-8b-local \
  --rag yes \
  --pipeline postedit \
  --tm \
  --experiment-suffix tm \
  --chap-start 7 \
  --chap-end 10 \
  --non-interactive 2>&1
echo "DONE"
