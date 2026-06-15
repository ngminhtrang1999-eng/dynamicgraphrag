#!/usr/bin/env bash
cd "/mnt/d/Dynamic GraphRAG"
export PATH="/home/macpham/miniconda3/envs/graphrag/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
for i in $(seq 1 48); do
  echo "=== attempt $i $(date) ==="
  python batch_judge_gemini.py --dir evaluation/alignment_results_qwen_rndfs_20_30 \
    --col-a RndFS --col-b GG --slug QwenRndFS \
    --chapters 20 21 22 23 24 25 26 27 28 29 30 \
    && { echo RNDFS_JUDGE_OK; exit 0; }
  echo "attempt $i failed; sleep 1800"
  sleep 1800
done
echo RNDFS_JUDGE_EXHAUSTED
