"""Compute LLM-judge vs human agreement from a filled spot-check CSV.

Run AFTER the user has filled the `your_choice` column in
evaluation/llm_judge/spot_check_for_review.csv.

Usage:
    python score_spot_check.py \\
        --review evaluation/llm_judge/spot_check_for_review.csv \\
        --key    evaluation/llm_judge/spot_check_key.csv \\
        --scores evaluation/llm_judge/scores_llama-3.1-8B_z-ai_glm-4.5-air_free.csv

Agreement = fraction of non-tie pairs where LLM-judge winner matches human winner.
Acceptance threshold for thesis: >= 0.70.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCORE_KEYS = ["accuracy", "fluency", "register", "cultural", "style"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--review", required=True, help="User-filled spot_check_for_review.csv")
    p.add_argument("--key", required=True, help="spot_check_key.csv generated alongside review")
    p.add_argument("--scores", required=True, help="LLM-judge scores CSV (batch_evaluate_LLM_judge output)")
    return p.parse_args()


def load_key(path):
    """Return dict {(chapter, sid): {A_is, B_is}}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["chapter"], r["sentence_id"])] = (r["A_is"], r["B_is"])
    return out


def load_scores(path):
    """Return dict {(chapter, sid, candidate_col): sum_of_5_axes}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                total = sum(int(r[k]) for k in SCORE_KEYS)
            except (ValueError, TypeError):
                continue
            out[(r["chapter"], r["sentence_id"], r["candidate_col"])] = total
    return out


def main():
    args = parse_args()
    key = load_key(args.key)
    scores = load_scores(args.scores)

    agree = 0
    disagree = 0
    judge_tie = 0
    human_tie = 0
    skipped_unscored = 0
    skipped_empty = 0
    details = []

    with open(args.review, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def normalize_choice(s):
        s = (s or "").strip().lower()
        # Accept variants users actually wrote: "a and b", "both", "a&b", "tie"
        if s in {"a and b", "a&b", "a & b", "both", "tie"}:
            return "tie"
        if s in {"a", "b"}:
            return s
        return s  # unknown form preserved for visibility

    for r in rows:
        chap = r["chapter"]
        sid = r["sentence_id"]
        human = normalize_choice(r.get("your_choice"))
        if not human:
            skipped_empty += 1
            continue
        if (chap, sid) not in key:
            skipped_unscored += 1
            continue
        a_is, b_is = key[(chap, sid)]
        a_score = scores.get((chap, sid, a_is))
        b_score = scores.get((chap, sid, b_is))
        if a_score is None or b_score is None:
            skipped_unscored += 1
            continue

        if a_score == b_score:
            judge_choice = "tie"
        elif a_score > b_score:
            judge_choice = "a"
        else:
            judge_choice = "b"

        if human == "tie":
            human_tie += 1
        if human == judge_choice:
            agree += 1
        else:
            disagree += 1

        details.append({
            "chapter": chap, "sentence_id": sid,
            "human": human, "judge": judge_choice,
            "A_is": a_is, "A_score": a_score,
            "B_is": b_is, "B_score": b_score,
            "match": "✓" if human == judge_choice else "✗",
        })

    total_compared = agree + disagree
    print("=" * 72)
    print(f"Spot-check results: {Path(args.review).name}")
    print("=" * 72)
    print(f"Total review rows:         {len(rows)}")
    print(f"Rows with answer filled:   {total_compared + (judge_tie if False else 0)}")
    print(f"Skipped (empty):           {skipped_empty}")
    print(f"Skipped (no judge score):  {skipped_unscored}")
    print(f"Human declared tie:        {human_tie}")
    print()
    print(f"Agreement: {agree}/{total_compared} = "
          f"{100 * agree / total_compared:.1f}%" if total_compared else "Agreement: n/a")
    print()
    print(f"{'chap':>4} {'sid':>4} {'A_is':<8} {'A_sc':>5} {'B_is':<8} {'B_sc':>5} "
          f"{'judge':<6} {'human':<6} match")
    print("-" * 72)
    for d in details:
        print(f"{d['chapter']:>4} {d['sentence_id']:>4} {d['A_is']:<8} {d['A_score']:>5} "
              f"{d['B_is']:<8} {d['B_score']:>5} {d['judge']:<6} {d['human']:<6} {d['match']}")

    if total_compared:
        rate = agree / total_compared
        print()
        if rate >= 0.70:
            print(f"✅ PASS — judge agreement {rate*100:.1f}% >= 70% threshold")
        else:
            print(f"⚠️  FAIL — judge agreement {rate*100:.1f}% < 70% threshold")
            print("    → consider tuning the rubric prompt or switching judge model.")
        return 0 if rate >= 0.70 else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
