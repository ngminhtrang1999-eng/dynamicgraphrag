"""Generate a blind pairwise spot-check CSV for human validation of the LLM-judge.

Reads an existing judge-scored CSV (output of batch_evaluate_LLM_judge.py),
finds sentence IDs where both compared candidates have scores, samples N pairs,
and writes a shuffled review CSV (user fills `your_choice`) plus a separate key CSV.

Usage:
    python generate_spot_check.py \\
        --scores evaluation/llm_judge/scores_llama-3.1-8B_z-ai_glm-4.5-air_free.csv \\
        --align  evaluation/alignment_results_llama-3.1-8B/aligned_master_chap_6.csv \\
        --pair-a RAG --pair-b No_RAG \\
        --n 15 \\
        --seed 42
"""

import argparse
import csv
import random
from pathlib import Path

REVIEW_COLS = [
    "chapter", "sentence_id", "source", "reference",
    "candidate_A", "candidate_B", "your_choice", "notes",
]
KEY_COLS = ["chapter", "sentence_id", "A_is", "B_is"]


def parse_args():
    p = argparse.ArgumentParser(description="Build pairwise spot-check CSV for human validation")
    p.add_argument("--scores", required=True, help="Judge-scored CSV from batch_evaluate_LLM_judge.py")
    p.add_argument("--align", required=True, help="Alignment CSV with source/ref/RAG/No_RAG/GG columns")
    p.add_argument("--pair-a", default="RAG", help="First candidate column to compare")
    p.add_argument("--pair-b", default="No_RAG", help="Second candidate column to compare")
    p.add_argument("--n", type=int, default=15, help="Number of pairs to sample (cap)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--out-dir", default="evaluation/llm_judge", help="Output directory")
    return p.parse_args()


def load_scored_sids(scores_path, cand_a, cand_b):
    """Return set of sentence_ids that have scores for BOTH compared candidates."""
    by_sid = {}
    with open(scores_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r["sentence_id"]
            cand = r["candidate_col"]
            by_sid.setdefault(sid, set()).add(cand)
    return {sid for sid, cands in by_sid.items() if cand_a in cands and cand_b in cands}


def load_alignment(align_path):
    """Return list of alignment rows keyed by Source_ID."""
    rows = {}
    with open(align_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sid = r.get("Source_ID")
            if sid is not None:
                rows[str(sid).strip()] = r
    return rows


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_sids = load_scored_sids(args.scores, args.pair_a, args.pair_b)
    align = load_alignment(args.align)

    # Restrict to rows present in both files and with non-empty candidates
    SKIP = {"<merged_above>", "<empty>", ""}
    candidates = []
    for sid in sorted(scored_sids, key=lambda s: int(s) if s.isdigit() else 9999):
        if sid not in align:
            continue
        row = align[sid]
        src = (row.get("Source_Text") or "").strip()
        a_text = (row.get(args.pair_a) or "").strip()
        b_text = (row.get(args.pair_b) or "").strip()
        if not src or src.lower() in SKIP or not a_text or not b_text:
            continue
        if a_text.lower() in SKIP or b_text.lower() in SKIP:
            continue
        candidates.append((sid, row))

    if not candidates:
        raise SystemExit("⚠️ No eligible sentence pairs after filtering.")

    sample = candidates if len(candidates) <= args.n else rng.sample(candidates, args.n)
    # Sort by sid for readable output
    sample.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 9999)

    # Infer chapter from filename
    chap_token = ""
    for token in Path(args.align).stem.split("_"):
        if token.startswith("chap"):
            chap_token = token.replace("chap", "")
        else:
            try:
                int(token)
                chap_token = token
            except ValueError:
                pass

    review_rows = []
    key_rows = []
    for sid, row in sample:
        flip = rng.random() < 0.5  # 50/50 which candidate is labeled A
        if flip:
            a_is, b_is = args.pair_b, args.pair_a
            cand_a_text = (row.get(args.pair_b) or "").strip()
            cand_b_text = (row.get(args.pair_a) or "").strip()
        else:
            a_is, b_is = args.pair_a, args.pair_b
            cand_a_text = (row.get(args.pair_a) or "").strip()
            cand_b_text = (row.get(args.pair_b) or "").strip()

        review_rows.append({
            "chapter": chap_token,
            "sentence_id": sid,
            "source": (row.get("Source_Text") or "").strip(),
            "reference": (row.get("ref") or "").strip(),
            "candidate_A": cand_a_text,
            "candidate_B": cand_b_text,
            "your_choice": "",
            "notes": "",
        })
        key_rows.append({
            "chapter": chap_token,
            "sentence_id": sid,
            "A_is": a_is,
            "B_is": b_is,
        })

    review_path = out_dir / "spot_check_for_review.csv"
    key_path = out_dir / "spot_check_key.csv"

    with open(review_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
        w.writeheader()
        w.writerows(review_rows)

    with open(key_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLS)
        w.writeheader()
        w.writerows(key_rows)

    print(f"✅ {len(review_rows)} pairs written.")
    print(f"   Review file (fill 'your_choice' column): {review_path}")
    print(f"   Key file (DO NOT open before review):    {key_path}")
    print()
    print("Instructions:")
    print("  - Open spot_check_for_review.csv in Excel/Sheets/LibreOffice.")
    print("  - For each row, read source + reference (optional) + candidate_A + candidate_B.")
    print("  - Fill 'your_choice' with: A | B | tie")
    print("  - Optional: add free-text 'notes' for context.")
    print("  - DO NOT look at spot_check_key.csv until you're done.")
    print("  - Save the CSV. Next session, run score_spot_check.py (or ask Claude) to compute agreement.")


if __name__ == "__main__":
    main()
