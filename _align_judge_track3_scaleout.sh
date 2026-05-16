#!/usr/bin/env bash
# Track 3 scale-out align + judge for chap 6, 8 (chap 7 already done in mini-test).
# Chap 9, 10 deferred pending Groq TPD reset.
#
# Per scale-out design: only RAG_BIG generated (no NORAG_BIG).
# align_ablation.py expects two candidates - pass RAG_BIG to both slots;
# we drop the duplicate after.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd '/mnt/d/Dynamic GraphRAG'

OUT_DIR=evaluation/alignment_results_track3
mkdir -p "$OUT_DIR"
TRANS_DIR=data/LOTM/translations

align_and_merge_gg() {
  local chap=$1
  local src_file="data/LOTM/chapters/chapter_$(printf '%04d' $chap).txt"
  local ref_file="data/LOTM/translations/reference/ref_${chap}.txt"
  local rag_big="$TRANS_DIR/trans_chap_${chap}_llama3.3-70b_yes_hygiene.txt"
  local gg_src="evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_${chap}_with_GG.csv"

  echo "=== Aligning chap $chap (RAG_BIG only; --tm-off slot duplicates --tm-on) ==="
  python align_ablation.py \
    --chap "$chap" \
    --src "$src_file" \
    --ref "$ref_file" \
    --tm-on  "$rag_big" \
    --tm-off "$rag_big" \
    --out-dir "$OUT_DIR" 2>&1 | tail -4

  # Drop dup TM_off, rename TM_on -> RAG_BIG, merge GG column
  python - <<EOF
import pandas as pd
p = "$OUT_DIR/aligned_master_chap_${chap}.csv"
df = pd.read_csv(p, encoding="utf-8-sig")
df = df.drop(columns=["TM_off"])
df = df.rename(columns={"TM_on": "RAG_BIG"})

gg = pd.read_csv("$gg_src", encoding="utf-8-sig")
merged = df.merge(gg[["Source_ID", "GG"]], on="Source_ID", how="left")
out = "$OUT_DIR/aligned_master_chap_${chap}_track3.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"chap ${chap} track3: n={len(merged)} cols={list(merged.columns)} -> {out}")
EOF
}

run_judge_vs_gg() {
  local chap=$1
  echo "=== chap $chap : RAG_BIG vs GG (deepseek-chat pairwise) ==="
  python batch_pairwise_judge.py \
    --review "$OUT_DIR/aligned_master_chap_${chap}_track3.csv" \
    --col-a RAG_BIG --col-b GG \
    --col-source Source_Text --col-ref ref --col-id Source_ID \
    --default-chapter "$chap" \
    --judge deepseek-chat \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --out "evaluation/llm_judge/pairwise_chap${chap}_Track3_RAGBIG_vs_GG_deepseek-chat.csv" \
    --sleep 1.0 2>&1 | tail -3
}

for chap in 6 8; do
  align_and_merge_gg $chap
done

echo ""
echo "=== Starting pairwise judge runs (deepseek-chat) ==="
for chap in 6 8; do
  run_judge_vs_gg $chap
done

echo ""
echo "ALL TRACK3 SCALE-OUT JUDGES (chap 6, 8) DONE"
echo "Output CSVs:"
ls -la evaluation/llm_judge/pairwise_chap{6,7,8}_Track3_RAGBIG_vs_GG_deepseek-chat.csv 2>&1
