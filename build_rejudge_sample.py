"""Build a stratified n≈500 re-judge sample of the chapters-6-10 evidence base.

For each of the 5 headline comparisons, sample ~20 already-judged pairs per
chapter (6-10), pull the candidate texts from the alignment masters, and write
one review CSV per comparison for batch_pairwise_judge.py (Gemini re-judge).
Deterministic: random.seed(13).

Output: evaluation/llm_judge/rejudge_<name>.csv
"""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent
EVAL = ROOT / "evaluation"
OUT = EVAL / "llm_judge"
PER_CHAPTER = 20
SKIP = {"<merged_above>", "<empty>", ""}

COMPARISONS = [
    {
        "name": "TMon_vs_TMoff",
        "dir": "alignment_results_qwen3-8b-local-ablation",
        "file": "aligned_master_chap_{n}_with_GG.csv",
        "fallback": "aligned_master_chap_{n}.csv",
        "col_a": "TM_on", "col_b": "TM_off",
        "old": ["pairwise_chap{n}_TMon_vs_TMoff_deepseek-chat.csv",
                "pairwise_ablation_chap{n}_TMon_vs_TMoff_deepseek-chat.csv"],
    },
    {
        "name": "Hygiene_vs_TMon",
        "dir": "alignment_results_qwen3-8b-local-hygiene",
        "file": "aligned_master_chap_{n}_hygiene.csv",
        "fallback": None,
        "col_a": "hygiene_on", "col_b": "TM_on",
        "old": ["pairwise_chap{n}_Hygiene_vs_TMon_deepseek-chat.csv"],
    },
    {
        "name": "A2_vs_Hygiene",
        "dir": "alignment_results_qwen3-8b-local-a2",
        "file": "aligned_master_chap_{n}_a2.csv",
        "fallback": None,
        "col_a": "a2_on", "col_b": "hygiene_on",
        "old": ["pairwise_chap{n}_A2_vs_Hygiene_deepseek-chat.csv"],
    },
    {
        "name": "A2_vs_GG",
        "dir": "alignment_results_qwen3-8b-local-a2",
        "file": "aligned_master_chap_{n}_a2_with_GG.csv",
        "fallback": None,
        "col_a": "a2_on", "col_b": "GG",
        "old": ["pairwise_chap{n}_A2_vs_GG_deepseek-chat.csv"],
    },
    {
        "name": "RAGBIG_vs_GG",
        "dir": "alignment_results_track3",
        "file": "aligned_master_chap_{n}_track3.csv",
        "fallback": None,
        "col_a": "RAG_BIG", "col_b": "GG",
        "old": ["pairwise_chap{n}_Track3_RAGBIG_vs_GG_deepseek-chat.csv"],
    },
]


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    random.seed(13)
    for spec in COMPARISONS:
        rows_out = []
        for n in range(6, 11):
            # already-judged ids for this comparison/chapter
            judged = None
            for pat in spec["old"]:
                p = OUT / pat.format(n=n)
                if p.exists():
                    judged = {r["sentence_id"] for r in read_csv(p)
                              if r.get("final_choice") in {"a", "b", "tie"}}
                    break
            if judged is None:
                print(f"[warn] no old judgments {spec['name']} chap {n}")
                continue
            ap = EVAL / spec["dir"] / spec["file"].format(n=n)
            if not ap.exists() and spec["fallback"]:
                ap = EVAL / spec["dir"] / spec["fallback"].format(n=n)
            if not ap.exists():
                print(f"[warn] no alignment file {ap}")
                continue
            cands = []
            for r in read_csv(ap):
                sid = r.get("Source_ID", "")
                a = (r.get(spec["col_a"]) or "").strip()
                b = (r.get(spec["col_b"]) or "").strip()
                if sid in judged and a.lower() not in SKIP and b.lower() not in SKIP:
                    cands.append({
                        "chapter": n, "sentence_id": sid,
                        "source": r.get("Source_Text", ""),
                        "reference": r.get("ref", ""),
                        "candidate_A": a, "candidate_B": b,
                    })
            random.shuffle(cands)
            rows_out.extend(cands[:PER_CHAPTER])
        out_path = OUT / f"rejudge_{spec['name']}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["chapter", "sentence_id", "source",
                                              "reference", "candidate_A", "candidate_B"])
            w.writeheader()
            w.writerows(rows_out)
        print(f"{spec['name']}: {len(rows_out)} pairs -> {out_path.name}")


if __name__ == "__main__":
    main()
