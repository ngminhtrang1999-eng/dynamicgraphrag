"""
LLM-judge evaluation runner for the B framework (literary MT eval).

Iterates over evaluation/alignment_results_<system>/aligned_master_chap_<N>.csv,
scores each candidate translation column (RAG / No_RAG / GG) on 5 literary axes
via an LLM judge (OpenRouter or any OpenAI-compatible endpoint).

Output: evaluation/llm_judge/scores_<system>_<judge>.csv with columns:
    chapter, sentence_id, candidate_col, accuracy, fluency, register, cultural,
    style, rationale, judge_model

Resume-capable: rows already scored (chapter, sid, candidate_col) are skipped.

Usage:
    python batch_evaluate_LLM_judge.py \\
        --judge deepseek/deepseek-chat-v3-0324:free \\
        --systems llama-3.1-8B \\
        --chapters 6 \\
        --candidates RAG,No_RAG,GG \\
        --n 30
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import const
from core.llm_backend import OpenAICompatLLM
from modules.prompts import build_literary_judge_prompt

EVAL_DIR = Path("evaluation")
ALIGN_PREFIX = "alignment_results_"
OUT_DIR = EVAL_DIR / "llm_judge"

DEFAULT_JUDGE = "deepseek/deepseek-chat-v3-0324:free"
DEFAULT_CANDIDATES = ["RAG", "No_RAG", "GG"]
SCORE_KEYS = ["accuracy", "fluency", "register", "cultural", "style"]
OUT_COLUMNS = (
    ["chapter", "sentence_id", "candidate_col"]
    + SCORE_KEYS
    + ["rationale", "judge_model"]
)

# Source rows whose Source_Text or ref equals one of these are skipped
# (the alignment process merges multi-sentence rows under a single anchor).
SKIP_VALUES = {"<merged_above>", "<empty>", ""}


def parse_args():
    p = argparse.ArgumentParser(description="LLM-judge runner for literary MT")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help="Model id (OpenRouter or other OpenAI-compat)")
    p.add_argument("--systems", required=True, help="Comma list of system names matching alignment_results_<NAME>")
    p.add_argument("--chapters", default="6", help="Comma list or 'all' (= 1..10)")
    p.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES),
                   help=f"CSV columns to score (default: {','.join(DEFAULT_CANDIDATES)})")
    p.add_argument("--n", type=int, default=0, help="Sample first N rows per chapter (0 = all)")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compat base URL (default OpenRouter)")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                   help="Env var name holding the API key (falls back to const.<name>)")
    p.add_argument("--sleep", type=float, default=2.0,
                   help="Seconds between calls (rate-limit guard for free tiers)")
    return p.parse_args()


def load_chapter_csv(system, chap):
    path = EVAL_DIR / f"{ALIGN_PREFIX}{system}" / f"aligned_master_chap_{chap}.csv"
    if not path.exists():
        return None
    # utf-8-sig handles the BOM at the top of these CSVs
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_existing(out_path):
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add((r["chapter"], r["sentence_id"], r["candidate_col"]))
    return done


def parse_judge_json(raw):
    """Strip code fences and parse the JSON object inside. Return dict or None."""
    if not raw:
        return None
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Find outermost JSON object
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def get_api_key(env_name):
    key = os.getenv(env_name, "")
    if key:
        return key
    return getattr(const, env_name, "")


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

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    if args.chapters == "all":
        chapters = list(range(1, 11))
    else:
        chapters = [int(c.strip()) for c in args.chapters.split(",")]

    judge_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.judge)

    for system in systems:
        out_path = OUT_DIR / f"scores_{system}_{judge_slug}.csv"
        done = load_existing(out_path)
        new_file = not out_path.exists()

        with open(out_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            if new_file:
                writer.writeheader()

            for chap in chapters:
                rows = load_chapter_csv(system, chap)
                if rows is None:
                    print(f"   ⚠️ No CSV for {system} chap {chap}")
                    continue
                if args.n > 0:
                    rows = rows[: args.n]

                for row in rows:
                    sid = str(row.get("Source_ID", "")).strip()
                    src = (row.get("Source_Text") or "").strip()
                    ref = (row.get("ref") or "").strip()

                    # Skip aligner artifacts
                    if not src or src.lower() in SKIP_VALUES:
                        continue
                    if ref.lower() in SKIP_VALUES:
                        ref = ""  # No reference for this row → switch judge to ref-free mode

                    for cand_col in candidates:
                        if (str(chap), sid, cand_col) in done:
                            continue
                        cand = (row.get(cand_col) or "").strip()
                        if not cand or cand.lower() in SKIP_VALUES:
                            continue

                        messages = build_literary_judge_prompt(
                            src, cand, reference_text=ref or None
                        )
                        try:
                            raw = judge.generate(messages, temperature=args.temperature)
                        except Exception as e:
                            print(f"   ❌ {system}/chap{chap}/sid{sid}/{cand_col}: API error: {e}")
                            continue

                        parsed = parse_judge_json(raw)
                        if not parsed or "scores" not in parsed:
                            print(f"   ⚠️ {system}/chap{chap}/sid{sid}/{cand_col}: bad JSON")
                            print(f"      raw[:200]: {raw[:200] if raw else '(empty)'}")
                            continue

                        scores = parsed.get("scores", {}) or {}
                        out_row = {
                            "chapter": chap,
                            "sentence_id": sid,
                            "candidate_col": cand_col,
                            "rationale": (parsed.get("rationale") or "")[:500],
                            "judge_model": args.judge,
                        }
                        for k in SCORE_KEYS:
                            out_row[k] = scores.get(k, "")

                        writer.writerow(out_row)
                        f.flush()

                        score_str = " ".join(f"{k[0]}={out_row[k]}" for k in SCORE_KEYS)
                        print(f"   ✓ {system}/chap{chap}/sid{sid}/{cand_col}: {score_str}")

                print(f"   ✅ Done {system} chap {chap}")
        print(f"📁 Saved: {out_path}")


if __name__ == "__main__":
    main()
