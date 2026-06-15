#!/usr/bin/env bash
# Align chapters 20-30: DeepSeek_RAG vs GG, and DeepSeek_zero vs GG,
# against the user-edited refs in chapters_vi/.
set -uo pipefail
cd "/mnt/d/Dynamic GraphRAG"
export PYTHONIOENCODING=utf-8

RAG_DIR=evaluation/alignment_results_deepseek_20_30
ZS_DIR=evaluation/alignment_results_deepseek_zeroshot_20_30

for N in 20 21 22 23 24 25 26 27 28 29 30; do
  SRC=$(printf 'data/LOTM/chapters/chapter_%04d.txt' "$N")
  GG=$(printf 'data/LOTM/translations/google_translate/translated_chapter_%04d.txt' "$N")
  REF="data/LOTM/chapters_vi/chap_${N}.txt"

  python align_ablation.py --chap "$N" --src "$SRC" --ref "$REF" \
    --tm-on  "data/LOTM/translations/trans_chap_${N}_deepseek-chat_yes_hygiene.txt" \
    --tm-off "$GG" --out-dir "$RAG_DIR" \
    && echo "[ok] RAG align chap $N" || echo "[FAIL] RAG align chap $N"

  python align_ablation.py --chap "$N" --src "$SRC" --ref "$REF" \
    --tm-on  "data/LOTM/translations/trans_chap_${N}_deepseek-chat_no.txt" \
    --tm-off "$GG" --out-dir "$ZS_DIR" \
    && echo "[ok] ZS  align chap $N" || echo "[FAIL] ZS  align chap $N"
done

python - <<'PY'
import pandas as pd
from pathlib import Path
for d, a in [("evaluation/alignment_results_deepseek_20_30", "DeepSeek"),
             ("evaluation/alignment_results_deepseek_zeroshot_20_30", "DeepSeek_ZS")]:
    for p in Path(d).glob("aligned_master_chap_*.csv"):
        df = pd.read_csv(p)
        df = df.rename(columns={"TM_on": a, "TM_off": "GG"})
        df.to_csv(p, index=False)
        print(f"renamed {p} -> cols {list(df.columns)}")
PY
echo DONE
