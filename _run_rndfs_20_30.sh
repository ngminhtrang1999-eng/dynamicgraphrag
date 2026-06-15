#!/usr/bin/env bash
# Random-few-shot baseline (reviewer requirement): Qwen3-8B full RAG pipeline
# but TM exemplars picked UNIFORMLY AT RANDOM (--tm-random) instead of by LaBSE
# similarity. Isolates the marginal value of dynamic retrieval over a static
# random few-shot prompt. Mirrors the A2 full-RAG run on chapters 20-30
# (test set B) in every other respect (a2-coref, hygiene, glossary, k=3).
#
# PREREQUISITE: Ollama must be running with qwen3:8b pulled (GPU; this is the
# heat-heavy step). KB dir qwen3-8b-local_yes_hygiene_a2on_rndfs is already
# seeded (copied from the a2on run: 928-entry TM + final glossary/graph).
#
# Run from WSL conda env graphrag.
set -uo pipefail
cd "/mnt/d/Dynamic GraphRAG"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

SUFFIX=hygiene_a2on_rndfs
TRANS_TAG=qwen3-8b-local_yes_${SUFFIX}
OUT_DIR=evaluation/alignment_results_qwen_rndfs_20_30
GEM_URL=https://generativelanguage.googleapis.com/v1beta/openai/
JUDGE=gemini-3.5-flash

# ---------- 1. translate (GPU, hours) ----------
python main.py \
  --model qwen3-8b-local --rag yes --pipeline postedit \
  --a2-coref --tm --tm-random --tm-k 3 \
  --chap-start 20 --chap-end 30 \
  --experiment-suffix "$SUFFIX" --non-interactive \
  && echo "[ok] random-few-shot translation done" \
  || { echo "[FAIL] translation"; exit 1; }

# ---------- 2. align vs GG against hand-made refs ----------
for N in 20 21 22 23 24 25 26 27 28 29 30; do
  SRC=$(printf 'data/LOTM/chapters/chapter_%04d.txt' "$N")
  GG=$(printf 'data/LOTM/translations/google_translate/translated_chapter_%04d.txt' "$N")
  REF="data/LOTM/chapters_vi/chap_${N}.txt"
  CAND="data/LOTM/translations/trans_chap_${N}_${TRANS_TAG}.txt"
  python align_ablation.py --chap "$N" --src "$SRC" --ref "$REF" \
    --tm-on "$CAND" --tm-off "$GG" --out-dir "$OUT_DIR" \
    && echo "[ok] align chap $N" || echo "[FAIL] align chap $N"
done

# rename candidate column TM_on -> RndFS, TM_off -> GG
python - <<'PY'
import pandas as pd
from pathlib import Path
for p in Path("evaluation/alignment_results_qwen_rndfs_20_30").glob("aligned_master_chap_*.csv"):
    df = pd.read_csv(p)
    if "TM_on" in df.columns:
        df = df.rename(columns={"TM_on": "RndFS", "TM_off": "GG"})
        df.to_csv(p, index=False)
print("column rename done")
PY

# ---------- 3. judge vs GG via Gemini Batch API (50% cost) ----------
python batch_judge_gemini.py \
  --dir "$OUT_DIR" --col-a RndFS --col-b GG --slug QwenRndFS \
  --chapters 20 21 22 23 24 25 26 27 28 29 30 \
  && echo "[ok] judge submitted/fetched" || echo "[FAIL] judge"

echo RNDFS_DONE
