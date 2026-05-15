#!/usr/bin/env bash
# Generates hygiene_on translations for chap 6-10 with:
#   - Cleaned glossary + TM (data/knowledge_base/qwen3-8b-local_yes_hygiene/)
#   - Hardened prompts (name policy + CJK/English hard rules in modules/prompts.py)
#   - Output filter prefers draft when post-edit is contaminated (modules/output_filter.py)
#
# Estimated runtime: ~75 min/chap × 5 = ~6.25h on qwen3:8b Ollama.
# Output: data/LOTM/translations/trans_chap_{6..10}_qwen3-8b-local_yes_hygiene.txt
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'
python main.py \
  --model qwen3-8b-local \
  --rag yes \
  --pipeline postedit \
  --tm \
  --experiment-suffix hygiene \
  --chap-start 6 \
  --chap-end 10 \
  --non-interactive 2>&1
echo "DONE"
