"""Score Phase 3 human responses (rubric-aligned framing) against the
deepseek-chat pairwise judge.

Two comparisons reported:

1. Set 1 (original 14 RAG vs No_RAG pairs):
   - NEW human choice (rubric-aligned, via review_app.py) vs deepseek-chat pairwise judge.
   - Also compares OLD human choice (prose-bias, from spot_check_for_review.csv) vs NEW,
     to quantify how much the framing changed the user's answers.

2. Set 2 (16 new TM_on vs TM_off pairs from chap 6 ablation):
   - Human choice vs deepseek-chat judge (anonymized A/B uses the key file to
     re-map both sides to TM_on / TM_off before comparing).

Headline question: did rubric-aligned framing push agreement past 70%?

Usage:
    python score_spot_check_v2.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "evaluation" / "llm_judge"

PAIRS = EVAL_DIR / "spot_check_v2_pairs.csv"
KEY = EVAL_DIR / "spot_check_v2_key.csv"
RESPONSES = EVAL_DIR / "spot_check_v2_responses.csv"

OLD_RAG_REVIEW = EVAL_DIR / "spot_check_for_review.csv"  # for set 1 old choice comparison

JUDGE_SET1 = EVAL_DIR / "pairwise_spot_check_for_review_deepseek-chat.csv"  # 14 pairs
JUDGE_SET2 = EVAL_DIR / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"  # chap 6 ablation


def normalize_choice(s):
    s = (s or "").strip().lower()
    if s in {"a and b", "a&b", "a & b", "both", "tie"}:
        return "tie"
    return s


def load_responses():
    """Return latest response per pair (last-write-wins on CSV row order)."""
    if not RESPONSES.exists():
        return {}
    out = {}
    with open(RESPONSES, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["set_name"], r["chapter"], r["sentence_id"])] = normalize_choice(r["choice"])
    return out


def load_key():
    """Return dict {(set_name, chap, sid): (A_is, B_is)}."""
    out = {}
    with open(KEY, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["set_name"], r["chapter"], r["sentence_id"])] = (r["A_is"], r["B_is"])
    return out


def load_judge_set1():
    """{(chap, sid): a/b/tie} from deepseek-chat pairwise judge on set 1."""
    out = {}
    with open(JUDGE_SET1, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["chapter"], r["sentence_id"])] = r["final_choice"]
    return out


def load_judge_set2():
    """{sid: a/b/tie} — chapter is fixed to 6 in this file."""
    out = {}
    with open(JUDGE_SET2, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["sentence_id"]] = r["final_choice"]
    return out


def load_old_set1_human():
    """Original (prose-bias) human choices on the 14 pairs."""
    out = {}
    with open(OLD_RAG_REVIEW, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["chapter"], r["sentence_id"])] = normalize_choice(r.get("your_choice"))
    return out


def fmt(rate, target=0.70):
    return f"{rate*100:.1f}% [{'PASS' if rate >= target else 'FAIL'} 70%]" if rate is not None else "n/a"


def score_set1(responses, judge_set1, old_human):
    """Compare NEW human (rubric) vs OLD human (prose-bias) vs deepseek judge.

    All three speak the same A/B labels (set 1 pairs are not re-shuffled), so
    we can compare directly.
    """
    print("=" * 78)
    print("SET 1 — Original 14 RAG vs No_RAG pairs (re-evaluated with rubric framing)")
    print("=" * 78)

    new_vs_judge_agree = old_vs_judge_agree = new_vs_old_agree = 0
    new_vs_judge_total = old_vs_judge_total = new_vs_old_total = 0
    rows = []

    for k in sorted(judge_set1.keys(), key=lambda x: int(x[1])):
        chap, sid = k
        judge = judge_set1[k]
        new = responses.get(("set1_rag_vs_norag", chap, sid))
        old = old_human.get((chap, sid))

        if new is not None:
            new_vs_judge_total += 1
            if new == judge:
                new_vs_judge_agree += 1
        if old is not None:
            old_vs_judge_total += 1
            if old == judge:
                old_vs_judge_agree += 1
        if new is not None and old is not None:
            new_vs_old_total += 1
            if new == old:
                new_vs_old_agree += 1

        rows.append((sid, judge, old or "-", new or "-"))

    print(f"{'sid':>4}  {'judge':<6} {'old(prose)':<11} {'new(rubric)':<12}")
    print("-" * 50)
    for sid, j, o, n in rows:
        print(f"{sid:>4}  {j:<6} {o:<11} {n:<12}")
    print()
    if new_vs_judge_total:
        rate = new_vs_judge_agree / new_vs_judge_total
        print(f"NEW (rubric) vs deepseek-chat: {new_vs_judge_agree}/{new_vs_judge_total} = {fmt(rate)}")
    else:
        print("NEW (rubric) vs deepseek-chat: no responses yet — run review_app.py")
    if old_vs_judge_total:
        rate = old_vs_judge_agree / old_vs_judge_total
        print(f"OLD (prose)  vs deepseek-chat: {old_vs_judge_agree}/{old_vs_judge_total} = {fmt(rate)}")
    if new_vs_old_total:
        rate = new_vs_old_agree / new_vs_old_total
        print(f"NEW vs OLD (how often user changed mind under rubric): {new_vs_old_agree}/{new_vs_old_total} agreed = {fmt(rate, target=0.5)}")
    print()


def score_set2(responses, judge_set2, key):
    """Compare human (anonymized A/B order) vs deepseek-chat judge (TM_on/TM_off).

    For each pair, both human's choice and judge's choice must be re-mapped to
    TM_on/TM_off using the key, then compared.
    """
    print("=" * 78)
    print("SET 2 — 16 new TM_on vs TM_off pairs from chap 6 ablation (anonymized)")
    print("=" * 78)

    agree = total = 0
    rows = []
    for k, (a_is, b_is) in key.items():
        set_name, chap, sid = k
        if set_name != "set2_tm_on_vs_off":
            continue
        human = responses.get(k)
        judge = judge_set2.get(sid)
        if not human or not judge:
            rows.append((sid, judge or "-", human or "-", "?", "?", "?"))
            continue

        # Map judge a/b to TM_on/TM_off
        # judge labels A/B refer to the ABLATION csv columns (always TM_on=A, TM_off=B)
        # — see JUDGE_SET2 source: pairwise_ablation_chap6_TMon_vs_TMoff has A=TM_on
        judge_pick = "tie"
        if judge == "a":
            judge_pick = "TM_on"
        elif judge == "b":
            judge_pick = "TM_off"

        # Map human a/b to TM_on/TM_off via key (which records what A and B truly are)
        human_pick = "tie"
        if human == "a":
            human_pick = a_is
        elif human == "b":
            human_pick = b_is

        total += 1
        if human_pick == judge_pick:
            agree += 1
        match = "OK" if human_pick == judge_pick else "no"
        rows.append((sid, judge, human, judge_pick, human_pick, match))

    print(f"{'sid':>4}  {'judge':<6} {'human':<6} {'judge_pick':<10} {'human_pick':<10} match")
    print("-" * 60)
    for sid, j, h, jp, hp, m in rows:
        print(f"{sid:>4}  {j:<6} {h:<6} {jp:<10} {hp:<10} {m}")
    print()
    if total:
        rate = agree / total
        print(f"Human vs deepseek-chat (after de-anonymizing): {agree}/{total} = {fmt(rate)}")
    else:
        print("No set-2 responses yet.")


def main():
    responses = load_responses()
    if not responses:
        print(f"No responses found at {RESPONSES}.")
        print("Run `python review_app.py` and review the 30 pairs in your browser first.")
        return 1

    key = load_key()
    judge_set1 = load_judge_set1()
    judge_set2 = load_judge_set2()
    old_set1_human = load_old_set1_human()

    print(f"Loaded {len(responses)} responses, {len(key)} keyed pairs.")
    print()
    score_set1(responses, judge_set1, old_set1_human)
    score_set2(responses, judge_set2, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
