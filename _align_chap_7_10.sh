#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'
for n in 7 8 9 10; do
  echo "=== chap $n ==="
  python align_ablation.py \
    --chap $n \
    --src "data/LOTM/chapters/chapter_$(printf %04d $n).txt" \
    --ref "data/LOTM/translations/reference/ref_${n}.txt" \
    --tm-on  "data/LOTM/translations/trans_chap_${n}_qwen3-8b-local_yes_tm.txt" \
    --tm-off "data/LOTM/translations/trans_chap_${n}_qwen3-8b-local_yes_notm.txt" \
    --out-dir evaluation/alignment_results_qwen3-8b-local-ablation 2>&1 | tail -4
done
echo "DONE"
