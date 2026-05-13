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
HEADLINE_KEYS = ["register", "fluency", "style"]
GATE_KEYS = ["accuracy", "cultural"]
GATE_MIN = 3  # any gate axis below this invalidates the candidate (prose rubric)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--review", required=True, help="User-filled spot_check_for_review.csv")
    p.add_argument("--key", required=True, help="spot_check_key.csv generated alongside review")
    p.add_argument("--scores", required=True, help="LLM-judge scores CSV (batch_evaluate_LLM_judge output)")
    p.add_argument(
        "--rubric",
        choices=["sum", "prose"],
        default="sum",
        help="sum: sum of 5 axes (original). prose: mean of register+fluency+style, "
             "with accuracy/cultural as gates (any axis <3 invalidates that candidate).",
    )
    p.add_argument(
        "--tie-threshold",
        type=float,
        default=0.0,
        help="If |a_score - b_score| <= threshold, judge declares tie. "
             "Default 0 (only exact equality is tie). Try 0.34 for prose (1/3 of one axis-point).",
    )
    return p.parse_args()


def load_key(path):
    """Return dict {(chapter, sid): {A_is, B_is}}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["chapter"], r["sentence_id"])] = (r["A_is"], r["B_is"])
    return out


def load_scores(path, rubric):
    """Return dict {(chapter, sid, candidate_col): score}.

    rubric=sum:  score = sum of 5 axes (integer 5-25).
    rubric=prose: score = mean of register+fluency+style (float 1.0-5.0); if any
        gate axis (accuracy/cultural) is below GATE_MIN, score = -inf to mark invalid.
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                axis = {k: int(r[k]) for k in SCORE_KEYS}
            except (ValueError, TypeError):
                continue
            if rubric == "sum":
                score = float(sum(axis.values()))
            else:  # prose
                gate_fail = any(axis[k] < GATE_MIN for k in GATE_KEYS)
                if gate_fail:
                    score = float("-inf")
                else:
                    score = sum(axis[k] for k in HEADLINE_KEYS) / len(HEADLINE_KEYS)
            out[(r["chapter"], r["sentence_id"], r["candidate_col"])] = score
    return out


def main():
    args = parse_args()
    key = load_key(args.key)
    scores = load_scores(args.scores, args.rubric)

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

        # Both invalid (gate fail) → tie. One invalid → the other wins.
        if a_score == float("-inf") and b_score == float("-inf"):
            judge_choice = "tie"
        elif a_score == float("-inf"):
            judge_choice = "b"
        elif b_score == float("-inf"):
            judge_choice = "a"
        elif abs(a_score - b_score) <= args.tie_threshold:
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
            "match": "OK" if human == judge_choice else "no",
        })

    total_compared = agree + disagree
    print("=" * 72)
    print(f"Spot-check results: {Path(args.review).name}")
    print(f"Rubric: {args.rubric}  |  tie_threshold: {args.tie_threshold}")
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
    print(f"{'chap':>4} {'sid':>4} {'A_is':<8} {'A_sc':>6} {'B_is':<8} {'B_sc':>6} "
          f"{'judge':<6} {'human':<6} match")
    print("-" * 72)

    def fmt_score(s):
        if s == float("-inf"):
            return "GATE"  # gate-failed candidate
        return f"{s:.2f}"

    for d in details:
        print(f"{d['chapter']:>4} {d['sentence_id']:>4} {d['A_is']:<8} {fmt_score(d['A_score']):>6} "
              f"{d['B_is']:<8} {fmt_score(d['B_score']):>6} {d['judge']:<6} {d['human']:<6} {d['match']}")

    if total_compared:
        rate = agree / total_compared
        print()
        if rate >= 0.70:
            print(f"[PASS] judge agreement {rate*100:.1f}% >= 70% threshold")
        else:
            print(f"[FAIL] judge agreement {rate*100:.1f}% < 70% threshold")
            print("    -> consider tuning the rubric prompt or switching judge model.")
        return 0 if rate >= 0.70 else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
