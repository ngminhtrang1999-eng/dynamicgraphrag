"""Prepare expanded spot-check (Phase 3) for human rubric-aligned evaluation.

Combines two sets of pairs:
1. Re-evaluation of the original 14 RAG vs No_RAG pairs (llama-3.1-8B chap 6).
   These are kept as-is in their original A/B assignment so we can compare the
   user's new (rubric-aligned) choice to the old (prose-bias) choice on the same
   data — measuring how much the framing matters.

2. New 16 pairs sampled from chap 6 ablation (TM_on vs TM_off), stratified by
   the deepseek-chat pairwise judge's verdict:
     - 6 pairs where judge picked A decisively (here A=TM_on)
     - 6 pairs where judge picked B decisively (here B=TM_off)
     - 4 pairs where judge said tie
   A/B order is RANDOMIZED per pair so the user can't infer which side is TM_on.
   The key file records the true mapping.

Outputs:
    evaluation/llm_judge/spot_check_v2_pairs.csv   (what the user sees)
    evaluation/llm_judge/spot_check_v2_key.csv     (true A/B identity per pair)

Random seed 42 for reproducibility.
"""

import csv
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "evaluation" / "llm_judge"

OLD_REVIEW = EVAL_DIR / "spot_check_for_review.csv"  # has source/ref/candidate_A/B
OLD_KEY = EVAL_DIR / "spot_check_key.csv"  # has A_is/B_is (RAG/No_RAG/GG)
ABLATION_CSV = ROOT / "evaluation" / "alignment_results_qwen3-8b-local-ablation" / "aligned_master_chap_6.csv"
JUDGE_PAIRWISE = EVAL_DIR / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"

OUT_PAIRS = EVAL_DIR / "spot_check_v2_pairs.csv"
OUT_KEY = EVAL_DIR / "spot_check_v2_key.csv"

SKIP_VALUES = {"<merged_above>", "<empty>", ""}


def load_set1():
    """Original 14 pairs, kept as-is (A/B not re-shuffled)."""
    rows = []
    with open(OLD_REVIEW, encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))
    with open(OLD_KEY, encoding="utf-8") as f:
        keymap = {(r["chapter"], r["sentence_id"]): (r["A_is"], r["B_is"])
                  for r in csv.DictReader(f)}

    for r in old_rows:
        chap = r["chapter"]
        sid = r["sentence_id"]
        a_is, b_is = keymap.get((chap, sid), ("?", "?"))
        rows.append({
            "set_name": "set1_rag_vs_norag",
            "chapter": chap,
            "sentence_id": sid,
            "source": r["source"],
            "reference": r["reference"],
            "candidate_A": r["candidate_A"],
            "candidate_B": r["candidate_B"],
            "A_is": a_is,
            "B_is": b_is,
        })
    return rows


def load_set2(n_a_win=6, n_b_win=6, n_tie=4):
    """Sample from chap 6 ablation, stratified by deepseek-chat verdict."""
    # Load ablation source: src/ref/TM_on/TM_off per Source_ID
    with open(ABLATION_CSV, encoding="utf-8-sig") as f:
        ablation = {r["Source_ID"]: r for r in csv.DictReader(f)}

    # Load judge verdicts
    judge = {}
    with open(JUDGE_PAIRWISE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            judge[r["sentence_id"]] = r["final_choice"]

    # Bucket by judge verdict, keep only rows that have valid non-merged candidates
    buckets = {"a": [], "b": [], "tie": []}
    for sid, verdict in judge.items():
        row = ablation.get(sid)
        if not row:
            continue
        src = (row.get("Source_Text") or "").strip()
        tm_on = (row.get("TM_on") or "").strip()
        tm_off = (row.get("TM_off") or "").strip()
        if not src or src.lower() in SKIP_VALUES:
            continue
        if not tm_on or tm_on.lower() in SKIP_VALUES:
            continue
        if not tm_off or tm_off.lower() in SKIP_VALUES:
            continue
        if verdict in buckets:
            buckets[verdict].append((sid, row))

    print(f"  bucket a (TM_on win): {len(buckets['a'])} available, sampling {n_a_win}")
    print(f"  bucket b (TM_off win): {len(buckets['b'])} available, sampling {n_b_win}")
    print(f"  bucket tie: {len(buckets['tie'])} available, sampling {n_tie}")

    sampled = (
        random.sample(buckets["a"], min(n_a_win, len(buckets["a"])))
        + random.sample(buckets["b"], min(n_b_win, len(buckets["b"])))
        + random.sample(buckets["tie"], min(n_tie, len(buckets["tie"])))
    )

    rows = []
    for sid, row in sampled:
        # Randomize A/B order so user can't tell which is TM_on
        if random.random() < 0.5:
            cand_a, cand_b = row["TM_on"], row["TM_off"]
            a_is, b_is = "TM_on", "TM_off"
        else:
            cand_a, cand_b = row["TM_off"], row["TM_on"]
            a_is, b_is = "TM_off", "TM_on"
        rows.append({
            "set_name": "set2_tm_on_vs_off",
            "chapter": "6",
            "sentence_id": sid,
            "source": row["Source_Text"],
            "reference": row["ref"],
            "candidate_A": cand_a,
            "candidate_B": cand_b,
            "A_is": a_is,
            "B_is": b_is,
        })
    return rows


def main():
    set1 = load_set1()
    print(f"Set 1 (original RAG vs No_RAG, re-eval): {len(set1)} pairs")

    set2 = load_set2()
    print(f"Set 2 (new TM_on vs TM_off, anonymized): {len(set2)} pairs")

    all_rows = set1 + set2
    print(f"Total pairs for Phase 3 review: {len(all_rows)}")

    # Output: user-facing file omits A_is/B_is to prevent peeking
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pair_cols = ["set_name", "chapter", "sentence_id",
                 "source", "reference", "candidate_A", "candidate_B"]
    with open(OUT_PAIRS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pair_cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in pair_cols})
    print(f"Saved pairs: {OUT_PAIRS}")

    # Key: separate file, do NOT open before reviewing
    key_cols = ["set_name", "chapter", "sentence_id", "A_is", "B_is"]
    with open(OUT_KEY, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=key_cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in key_cols})
    print(f"Saved key:   {OUT_KEY}")


if __name__ == "__main__":
    main()
