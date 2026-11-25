#!/usr/bin/env bash
# Align A2_on translations against the existing hygiene baseline for chap 6-10.
# Uses align_ablation.py with --tm-on=a2on file and --tm-off=hygiene file,
# then renames the output columns TM_on/TM_off → a2_on/hygiene_on.
#
# Produces aligned_master_chap_{6..10}_a2.csv with columns:
#   Source_ID, Source_Text, ref, a2_on, hygiene_on
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrag
cd "$(dirname "$0")"

OUT_DIR=evaluation/alignment_results_qwen3-8b-local-a2
mkdir -p "$OUT_DIR"
TRANS_DIR=data/LOTM/translations

for n in 6 7 8 9 10; do
  echo "=== chap $n ==="
  python align_ablation.py \
    --chap $n \
    --src "data/LOTM/chapters/chapter_$(printf %04d $n).txt" \
    --ref "data/LOTM/translations/reference/ref_${n}.txt" \
    --tm-on  "$TRANS_DIR/trans_chap_${n}_qwen3-8b-local_yes_hygiene_a2on.txt" \
    --tm-off "$TRANS_DIR/trans_chap_${n}_qwen3-8b-local_yes_hygiene.txt" \
    --out-dir "$OUT_DIR" 2>&1 | tail -4

  python - <<EOF
import pandas as pd
p = "$OUT_DIR/aligned_master_chap_${n}.csv"
df = pd.read_csv(p, encoding="utf-8-sig")
df = df.rename(columns={"TM_on": "a2_on", "TM_off": "hygiene_on"})
out = p.replace("aligned_master_chap_${n}.csv", "aligned_master_chap_${n}_a2.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print("renamed cols ->", list(df.columns), "wrote", out)
EOF
done
echo "DONE"
