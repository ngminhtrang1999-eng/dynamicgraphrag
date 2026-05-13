"""Pairwise LLM-judge runner for the B framework (literary MT eval).

Reads a spot-check CSV (chapter, sentence_id, source, reference, candidate_A,
candidate_B, your_choice, notes), asks the judge "which would a Vietnamese
reader prefer", and writes pairwise outcomes with position-bias control.

Position-bias control: each pair is judged twice — once in original (A, B)
order, once swapped (B, A). The final choice is decisive only if both calls
agree (after un-swapping the second). Otherwise the pair is recorded as "tie".

Output: evaluation/llm_judge/pairwise_<system>_<judge>.csv with columns:
    chapter, sentence_id, call_1_choice, call_2_choice_unswapped, final_choice,
    rationale_1, rationale_2, judge_model

Usage:
    python batch_pairwise_judge.py \\
        --review evaluation/llm_judge/spot_check_for_review.csv \\
        --judge z-ai/glm-4.5-air:free
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import const
from core.llm_backend import OpenAICompatLLM
from modules.prompts import build_literary_pairwise_judge_prompt

DEFAULT_JUDGE = "z-ai/glm-4.5-air:free"
OUT_DIR = Path("evaluation/llm_judge")


def parse_args():
    p = argparse.ArgumentParser(description="Pairwise LLM-judge for literary MT")
    p.add_argument("--review", required=True, help="Input CSV (spot-check or aligned_master_chap_N.csv)")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help="OpenAI-compat model id")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds between API calls")
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: evaluation/llm_judge/pairwise_<review_stem>_<judge_slug>.csv",
    )
    p.add_argument("--n", type=int, default=0, help="First N rows only (0 = all). Useful for smoke tests.")
    # Column mapping — defaults match spot_check_for_review.csv format
    p.add_argument("--col-a", default="candidate_A", help="CSV column for candidate A")
    p.add_argument("--col-b", default="candidate_B", help="CSV column for candidate B")
    p.add_argument("--col-source", default="source", help="CSV column for source text")
    p.add_argument("--col-ref", default="reference", help="CSV column for reference (gold); empty col -> ref-free")
    p.add_argument("--col-id", default="sentence_id", help="CSV column for sentence id (per-chapter)")
    p.add_argument("--col-chapter", default="chapter", help="CSV column for chapter id (optional)")
    p.add_argument("--default-chapter", default="", help="Used if --col-chapter is missing in the CSV")
    return p.parse_args()


def get_api_key(env_name):
    return os.getenv(env_name, "") or getattr(const, env_name, "")


def parse_choice_json(raw):
    """Parse {"choice": "A|B|tie", "rationale": "..."}. Return (choice, rationale)."""
    if not raw:
        return None, ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None, raw[:200]
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, raw[:200]

    choice = (obj.get("choice") or "").strip().lower()
    rationale = (obj.get("rationale") or "")[:500]
    if choice in {"a", "b", "tie"}:
        return choice, rationale
    return None, rationale


def unswap(choice):
    """Map a swapped call back to original A/B labels."""
    if choice == "a":
        return "b"
    if choice == "b":
        return "a"
    return choice  # tie / None


def reconcile(c1, c2_unswapped):
    """Combine two calls. Decisive only if both agree on A or B; else tie."""
    if c1 is None or c2_unswapped is None:
        return None
    if c1 == c2_unswapped and c1 in {"a", "b"}:
        return c1
    if c1 == "tie" and c2_unswapped == "tie":
        return "tie"
    return "tie"  # mixed → not robust, declare tie


def load_existing(path):
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add((r["chapter"], r["sentence_id"]))
    return done


SKIP_VALUES = {"<merged_above>", "<empty>", ""}


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key(args.api_key_env)
    if not api_key:
        raise SystemExit(f"⚠️ No API key in env {args.api_key_env} or const.")

    judge = OpenAICompatLLM(
        model_name=args.judge,
        base_url=args.base_url,
        api_key=api_key,
        rate_limit_sleep=args.sleep,
    )

    review_path = Path(args.review)
    if args.out:
        out_path = Path(args.out)
    else:
        judge_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.judge)
        out_path = OUT_DIR / f"pairwise_{review_path.stem}_{judge_slug}.csv"

    done = load_existing(out_path)
    new_file = not out_path.exists()

    # utf-8-sig handles the BOM at the top of aligned_master_chap_*.csv files
    with open(review_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if args.n > 0:
        rows = rows[: args.n]

    out_cols = [
        "chapter", "sentence_id",
        "call_1_choice", "call_2_choice_unswapped", "final_choice",
        "rationale_1", "rationale_2", "judge_model",
    ]

    with open(out_path, "a", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        if new_file:
            writer.writeheader()

        for idx, r in enumerate(rows):
            chap = str(r.get(args.col_chapter, args.default_chapter) or args.default_chapter).strip()
            sid = str(r.get(args.col_id, "")).strip()
            if (chap, sid) in done:
                print(f"   = skip pair {chap}/{sid} (already judged)")
                continue

            src = (r.get(args.col_source) or "").strip()
            ref = (r.get(args.col_ref) or "").strip()
            cand_a = (r.get(args.col_a) or "").strip()
            cand_b = (r.get(args.col_b) or "").strip()

            # Skip aligner artifacts on source / reference
            if not src or src.lower() in SKIP_VALUES:
                continue
            if ref.lower() in SKIP_VALUES:
                ref = ""  # <MERGED_ABOVE> rows lose their reference → ref-free mode

            if not cand_a or cand_a.lower() in SKIP_VALUES:
                print(f"   ! pair {chap}/{sid}: candidate A empty/merged, skipped")
                continue
            if not cand_b or cand_b.lower() in SKIP_VALUES:
                print(f"   ! pair {chap}/{sid}: candidate B empty/merged, skipped")
                continue

            # Call 1 — original order
            msgs_1 = build_literary_pairwise_judge_prompt(src, cand_a, cand_b, reference_text=ref or None)
            try:
                raw_1 = judge.generate(msgs_1, temperature=args.temperature)
            except Exception as e:
                print(f"   ! pair {chap}/{sid} call 1 error: {e}")
                continue
            c1, rationale_1 = parse_choice_json(raw_1)

            # Call 2 — swapped
            msgs_2 = build_literary_pairwise_judge_prompt(src, cand_b, cand_a, reference_text=ref or None)
            try:
                raw_2 = judge.generate(msgs_2, temperature=args.temperature)
            except Exception as e:
                print(f"   ! pair {chap}/{sid} call 2 error: {e}")
                continue
            c2_raw, rationale_2 = parse_choice_json(raw_2)
            c2_unswapped = unswap(c2_raw)

            final = reconcile(c1, c2_unswapped)

            out_row = {
                "chapter": chap,
                "sentence_id": sid,
                "call_1_choice": c1 if c1 else "PARSE_ERR",
                "call_2_choice_unswapped": c2_unswapped if c2_unswapped else "PARSE_ERR",
                "final_choice": final if final else "PARSE_ERR",
                "rationale_1": rationale_1,
                "rationale_2": rationale_2,
                "judge_model": args.judge,
            }
            writer.writerow(out_row)
            f_out.flush()

            print(f"   [{idx+1}/{len(rows)}] pair {chap}/{sid}: "
                  f"c1={c1} c2={c2_unswapped} -> final={final}")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
