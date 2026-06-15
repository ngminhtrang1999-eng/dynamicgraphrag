#!/usr/bin/env bash
# Round 1: align with FINAL hand-made refs, then Gemini-judge everything ready.
# Comparisons: DeepSeek_RAG vs GG (20-30), DeepSeek_ZS vs GG (6-10, 20-30),
#              Qwen_ZS vs GG (6-10, 20-30).
# Qwen A2 (20-30) is judged in round 2 when its translation chain finishes.
set -uo pipefail
cd "/mnt/d/Dynamic GraphRAG"
export PYTHONIOENCODING=utf-8

GEM_URL=https://generativelanguage.googleapis.com/v1beta/openai/
JUDGE=gemini-3.5-flash

align() { # chap src ref cand gg outdir
  if [ -f "$6/aligned_master_chap_$1.csv" ]; then
    echo "[skip] align $6 chap $1 (exists)"
    return 0
  fi
  python align_ablation.py --chap "$1" --src "$2" --ref "$3" \
    --tm-on "$4" --tm-off "$5" --out-dir "$6" \
    && echo "[ok] align $6 chap $1" || echo "[FAIL] align $6 chap $1"
}

ref_for() { # chapter number -> ref path
  if [ "$1" -le 10 ]; then
    echo "data/LOTM/translations/reference/ref_$1.txt"
  else
    echo "data/LOTM/chapters_vi/chap_$1.txt"
  fi
}

# ---------- alignment ----------
for N in 20 21 22 23 24 25 26 27 28 29 30; do
  SRC=$(printf 'data/LOTM/chapters/chapter_%04d.txt' "$N")
  GG=$(printf 'data/LOTM/translations/google_translate/translated_chapter_%04d.txt' "$N")
  REF=$(ref_for "$N")
  align "$N" "$SRC" "$REF" "data/LOTM/translations/trans_chap_${N}_deepseek-chat_yes_hygiene.txt" "$GG" evaluation/alignment_results_deepseek_20_30
  align "$N" "$SRC" "$REF" "data/LOTM/translations/trans_chap_${N}_deepseek-chat_no.txt" "$GG" evaluation/alignment_results_deepseek_zs
  align "$N" "$SRC" "$REF" "data/LOTM/translations/trans_chap_${N}_qwen3-8b-local_no.txt" "$GG" evaluation/alignment_results_qwen_zs
done
for N in 6 7 8 9 10; do
  SRC=$(printf 'data/LOTM/chapters/chapter_%04d.txt' "$N")
  GG=$(printf 'data/LOTM/translations/google_translate/translated_chapter_%04d.txt' "$N")
  REF=$(ref_for "$N")
  align "$N" "$SRC" "$REF" "data/LOTM/translations/trans_chap_${N}_deepseek-chat_no.txt" "$GG" evaluation/alignment_results_deepseek_zs
  align "$N" "$SRC" "$REF" "data/LOTM/translations/trans_chap_${N}_qwen3-8b-local_no.txt" "$GG" evaluation/alignment_results_qwen_zs
done

# rename candidate columns per dir
python - <<'PY'
import pandas as pd
from pathlib import Path
for d, a in [("evaluation/alignment_results_deepseek_20_30", "DeepSeek"),
             ("evaluation/alignment_results_deepseek_zs", "DeepSeek_ZS"),
             ("evaluation/alignment_results_qwen_zs", "Qwen_ZS")]:
    for p in Path(d).glob("aligned_master_chap_*.csv"):
        df = pd.read_csv(p)
        if "TM_on" in df.columns:
            df = df.rename(columns={"TM_on": a, "TM_off": "GG"})
            df.to_csv(p, index=False)
print("column rename done")
PY

# ---------- judging ----------
judge() { # csv col_a chapter outslug
  python batch_pairwise_judge.py --review "$1" \
    --col-a "$2" --col-b GG --col-source Source_Text --col-ref ref \
    --col-id Source_ID --default-chapter "$3" \
    --judge "$JUDGE" --base-url "$GEM_URL" --api-key-env GEMINI_API_KEY \
    --sleep 0.4 \
    --out "evaluation/llm_judge/pairwise_chap$3_$4_vs_GG_gemini35flash.csv" \
    && echo "[ok] judge $4 chap $3" || echo "[FAIL] judge $4 chap $3"
}

for N in 20 21 22 23 24 25 26 27 28 29 30; do
  judge "evaluation/alignment_results_deepseek_20_30/aligned_master_chap_${N}.csv" DeepSeek "$N" DeepSeek
done
for N in 6 7 8 9 10 20 21 22 23 24 25 26 27 28 29 30; do
  judge "evaluation/alignment_results_deepseek_zs/aligned_master_chap_${N}.csv" DeepSeek_ZS "$N" DeepSeekZS
done
for N in 6 7 8 9 10 20 21 22 23 24 25 26 27 28 29 30; do
  judge "evaluation/alignment_results_qwen_zs/aligned_master_chap_${N}.csv" Qwen_ZS "$N" QwenZS
done

echo ROUND1_DONE
