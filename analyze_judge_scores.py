"""Summarize llm-judge score distributions from batch_evaluate_LLM_judge.py output.

Usage:
    python analyze_judge_scores.py [path_to_scores.csv]
    # defaults to scores_llama-3.1-8B_z-ai_glm-4.5-air_free.csv if no arg.

Prints per-(candidate, axis) mean / std / count / distribution and overall
mean per candidate. Useful for sanity-checking that the judge discriminates
across systems before committing to a full evaluation run.
"""
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "evaluation/llm_judge/scores_llama-3.1-8B_z-ai_glm-4.5-air_free.csv")

scores = defaultdict(list)
total = 0
with open(path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        total += 1
        cand = row["candidate_col"]
        for axis in ["accuracy", "fluency", "register", "cultural", "style"]:
            try:
                scores[(cand, axis)].append(int(row[axis]))
            except (ValueError, TypeError):
                pass

print(f"File: {path.name}")
print(f"Total rows scored: {total}")
print()
hdr = f"{'candidate':<8} {'axis':<10} {'mean':>5} {'std':>5} {'n':>3}  distribution"
print(hdr)
print("-" * len(hdr) * 2)
for cand in ["RAG", "No_RAG", "GG"]:
    for axis in ["accuracy", "fluency", "register", "cultural", "style"]:
        vals = scores[(cand, axis)]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0
        dist = dict(sorted(Counter(vals).items()))
        print(f"{cand:<8} {axis:<10} {mean:>5.2f} {std:>5.2f} {len(vals):>3}  {dist}")
    print()

print("=== Overall mean per candidate (across all 5 axes) ===")
for cand in ["RAG", "No_RAG", "GG"]:
    all_vals = []
    for axis in ["accuracy", "fluency", "register", "cultural", "style"]:
        all_vals.extend(scores[(cand, axis)])
    if all_vals:
        m = statistics.mean(all_vals)
        s = statistics.stdev(all_vals) if len(all_vals) > 1 else 0
        print(f"  {cand:<8}  mean={m:.2f}  std={s:.2f}  n={len(all_vals)}")
