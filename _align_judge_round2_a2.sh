#!/usr/bin/env bash
# Round 2: align Qwen A2 (20-30) vs GG now; judge after the Gemini daily-quota
# reset (sleep gate set from launch time).
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

GEM_URL=https://generativelanguage.googleapis.com/v1beta/openai/
JUDGE=gemini-3.5-flash
DIR=evaluation/alignment_results_a2_20_30

for N in 20 21 22 23 24 25 26 27 28 29 30; do
  if [ -f "$DIR/aligned_master_chap_${N}.csv" ]; then
    echo "[skip] align chap $N"
    continue
  fi
  SRC=$(printf 'data/LOTM/chapters/chapter_%04d.txt' "$N")
  GG=$(printf 'data/LOTM/translations/google_translate/translated_chapter_%04d.txt' "$N")
  python align_ablation.py --chap "$N" --src "$SRC" --ref "data/LOTM/chapters_vi/chap_${N}.txt" \
    --tm-on "data/LOTM/translations/trans_chap_${N}_qwen3-8b-local_yes_hygiene_a2on.txt" \
    --tm-off "$GG" --out-dir "$DIR" \
    && echo "[ok] align chap $N" || echo "[FAIL] align chap $N"
done

python - <<'PY'
import pandas as pd
from pathlib import Path
for p in Path("evaluation/alignment_results_a2_20_30").glob("aligned_master_chap_*.csv"):
    df = pd.read_csv(p)
    if "TM_on" in df.columns:
        df.rename(columns={"TM_on": "A2", "TM_off": "GG"}).to_csv(p, index=False)
print("rename done")
PY

echo "sleeping until Gemini daily quota reset..."
sleep "${QUOTA_SLEEP:-10200}"

for N in 20 21 22 23 24 25 26 27 28 29 30; do
  python batch_pairwise_judge.py \
    --review "$DIR/aligned_master_chap_${N}.csv" \
    --col-a A2 --col-b GG --col-source Source_Text --col-ref ref \
    --col-id Source_ID --default-chapter "$N" \
    --judge "$JUDGE" --base-url "$GEM_URL" --api-key-env GEMINI_API_KEY \
    --sleep 0.4 \
    --out "evaluation/llm_judge/pairwise_chap${N}_A2new_vs_GG_gemini35flash.csv" \
    && echo "[ok] judge A2 chap $N" || echo "[FAIL] judge A2 chap $N"
done
echo ROUND2_DONE
