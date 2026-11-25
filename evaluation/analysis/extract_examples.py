"""Track 1 step 4: pull worked examples for the qualitative taxonomy.

For each comparison (TMon_vs_GG, TMon_vs_TMoff), sample N rows of each
final_choice outcome (a-wins, b-wins) and join with aligned_master_chap_X.csv
to attach Source_Text + ref + TM_on + TM_off + GG. Prints + dumps to JSON.

Sampling strategy: deterministic — pick first N decisive rows per chapter,
then concatenate. Avoids bias from random.seed drift across runs.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = ROOT / "evaluation" / "llm_judge"
ALIGN_DIR = ROOT / "evaluation" / "alignment_results_qwen3-8b-local-ablation"
OUT_DIR = ROOT / "evaluation" / "analysis"

# A=TM_on for both comparisons; B differs.
COMPARISONS = [
    {
        "label": "TMon_vs_GG",
        "b_col": "GG",
        "csvs": [JUDGE_DIR / f"pairwise_chap{c}_TMon_vs_GG_deepseek-chat.csv" for c in range(6, 11)],
    },
    {
        "label": "TMon_vs_TMoff",
        "b_col": "TM_off",
        "csvs": [JUDGE_DIR / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"]
                + [JUDGE_DIR / f"pairwise_chap{c}_TMon_vs_TMoff_deepseek-chat.csv" for c in range(7, 11)],
    },
]

SAMPLES_PER_OUTCOME_PER_CHAP = 3  # 3 a-wins + 3 b-wins per chapter per comparison


def load_aligned(chap):
    p = ALIGN_DIR / f"aligned_master_chap_{chap}_with_GG.csv"
    out = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["Source_ID"]] = row
    return out


def main():
    for comp in COMPARISONS:
        examples = {"a_wins": [], "b_wins": []}
        for csv_path in comp["csvs"]:
            chap = csv_path.stem.split("chap")[-1].split("_")[0]
            aligned = load_aligned(chap)
            per_outcome_taken = {"a": 0, "b": 0}
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fc = row["final_choice"].strip().lower()
                    if fc not in ("a", "b"):
                        continue
                    if per_outcome_taken[fc] >= SAMPLES_PER_OUTCOME_PER_CHAP:
                        continue
                    sid = row["sentence_id"]
                    align = aligned.get(sid)
                    if not align:
                        continue
                    example = {
                        "chapter": chap,
                        "sentence_id": sid,
                        "source": align["Source_Text"],
                        "ref": align["ref"],
                        "A_TM_on": align["TM_on"],
                        "B": align[comp["b_col"]],
                        "B_label": comp["b_col"],
                        "final_choice": fc,
                        "call_1": row["call_1_choice"],
                        "call_2": row["call_2_choice_unswapped"],
                        "rationale_1": row["rationale_1"],
                        "rationale_2": row["rationale_2"],
                    }
                    key = "a_wins" if fc == "a" else "b_wins"
                    examples[key].append(example)
                    per_outcome_taken[fc] += 1
        out_path = OUT_DIR / f"examples_{comp['label']}.json"
        out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{comp['label']}: a_wins={len(examples['a_wins'])} b_wins={len(examples['b_wins'])} -> {out_path}")


if __name__ == "__main__":
    main()
