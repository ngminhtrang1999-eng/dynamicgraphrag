"""Build an expanded blind human-evaluation kit for the literary-MT paper.

Addresses the reviewer requirement: expand human validation from n=30 (one
rater, chapter 6 only) to n>=100 across multiple raters, with chance-corrected
inter-rater agreement (Cohen's / Fleiss' kappa, computed by score_human_eval.py).

Design
------
Three headline comparisons from test set B (chapters 20--30), each judged
already by the independent Gemini 3.5 Flash judge:

    DeepSeek-V3.1 + RAG  vs GG   (the frontier inversion, 63.1% decisive)
    Qwen3-8B A2 full RAG vs GG   (the 8B pipeline,        27.7% decisive)
    DeepSeek-V3.1 ZS     vs GG   (zero-shot baseline,     60.5% decisive)

For each comparison, pairs are sampled *stratified by the LLM judge's verdict*
(decisive-A, decisive-B, tie) so the human sheet spans the full agreement space,
not just the easy cases. A/B order is randomized per pair and the system
identity is withheld from the rater-facing file; the key file records the truth.

Outputs (evaluation/human_eval/):
    human_eval_pairs.csv   -- blind sheet given to each rater (empty `choice`)
    human_eval_key.csv     -- true A/B identity + original judge verdict

Each rater fills `choice` in {A, B, tie} on a COPY named
    human_eval_rater_<name>.csv
and score_human_eval.py consumes those copies.

Reproducible: random seed 42.
"""

import csv
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).parent
EVAL = ROOT / "evaluation"
JUDGE_DIR = EVAL / "llm_judge"
OUT_DIR = EVAL / "human_eval"

CHAPTERS = range(20, 31)
SKIP = {"<merged_above>", "<empty>", ""}

# (label, judge-slug, alignment dir, candidate column)
COMPARISONS = [
    ("deepseek_rag_vs_gg", "DeepSeek",   "alignment_results_deepseek_20_30", "DeepSeek"),
    ("qwen_a2_vs_gg",      "A2new",      "alignment_results_a2_20_30",       "A2"),
    ("deepseek_zs_vs_gg",  "DeepSeekZS", "alignment_results_deepseek_zs",    "DeepSeek_ZS"),
]

# per comparison: how many to sample from each judge-verdict bucket
N_A, N_B, N_TIE = 16, 16, 8  # 40 per comparison, 120 total


def load_judge_verdicts(slug):
    """sid-key 'chap|sid' -> final_choice for one comparison across chapters."""
    verdicts = {}
    for c in CHAPTERS:
        f = JUDGE_DIR / f"pairwise_chap{c}_{slug}_vs_GG_gemini35flash.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                fc = (r.get("final_choice") or "").strip().lower()
                if fc in {"a", "b", "tie"}:
                    verdicts[f"{r['chapter']}|{r['sentence_id']}"] = fc
    return verdicts


def load_aligned(dir_name, cand_col):
    """'chap|sid' -> dict(source, ref, cand, gg) for valid, non-merged rows."""
    rows = {}
    for c in CHAPTERS:
        f = EVAL / dir_name / f"aligned_master_chap_{c}.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                sid = str(r.get("Source_ID", "")).strip()
                src = (r.get("Source_Text") or "").strip()
                ref = (r.get("ref") or "").strip()
                cand = (r.get(cand_col) or "").strip()
                gg = (r.get("GG") or "").strip()
                if not src or src.lower() in SKIP:
                    continue
                if not cand or cand.lower() in SKIP:
                    continue
                if not gg or gg.lower() in SKIP:
                    continue
                rows[f"{c}|{sid}"] = {
                    "source": src,
                    "reference": ref if ref.lower() not in SKIP else "",
                    "cand": cand,
                    "gg": gg,
                }
    return rows


def sample_comparison(label, slug, dir_name, cand_col):
    verdicts = load_judge_verdicts(slug)
    aligned = load_aligned(dir_name, cand_col)

    buckets = {"a": [], "b": [], "tie": []}
    for key, fc in verdicts.items():
        if key in aligned:
            buckets[fc].append(key)

    print(f"[{label}] available -> a:{len(buckets['a'])} b:{len(buckets['b'])} "
          f"tie:{len(buckets['tie'])}")

    picked = (
        random.sample(buckets["a"], min(N_A, len(buckets["a"])))
        + random.sample(buckets["b"], min(N_B, len(buckets["b"])))
        + random.sample(buckets["tie"], min(N_TIE, len(buckets["tie"])))
    )

    pairs, keys = [], []
    for key in picked:
        chap, sid = key.split("|", 1)
        a = aligned[key]
        # candidate is system A in judge convention; randomize for the human
        if random.random() < 0.5:
            cand_a, cand_b = a["cand"], a["gg"]
            a_is, b_is = "CAND", "GG"
        else:
            cand_a, cand_b = a["gg"], a["cand"]
            a_is, b_is = "GG", "CAND"
        pairs.append({
            "comparison": label,
            "chapter": chap,
            "sentence_id": sid,
            "source": a["source"],
            "reference": a["reference"],
            "candidate_A": cand_a,
            "candidate_B": cand_b,
            "choice": "",  # rater fills: A / B / tie
        })
        keys.append({
            "comparison": label,
            "chapter": chap,
            "sentence_id": sid,
            "A_is": a_is,
            "B_is": b_is,
            "candidate_system": cand_col,
            "judge_verdict": verdicts[key],  # in candidate-vs-GG convention (a/b/tie)
        })
    return pairs, keys


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs, all_keys = [], []
    for label, slug, dir_name, cand_col in COMPARISONS:
        p, k = sample_comparison(label, slug, dir_name, cand_col)
        all_pairs.extend(p)
        all_keys.extend(k)

    # Shuffle the combined deck so consecutive items aren't same-comparison
    combined = list(zip(all_pairs, all_keys))
    random.shuffle(combined)
    all_pairs, all_keys = [list(t) for t in zip(*combined)]

    # global blind item ids
    for i, (p, k) in enumerate(zip(all_pairs, all_keys), 1):
        p["item_id"] = i
        k["item_id"] = i

    pair_cols = ["item_id", "comparison", "chapter", "sentence_id",
                 "source", "reference", "candidate_A", "candidate_B", "choice"]
    key_cols = ["item_id", "comparison", "chapter", "sentence_id",
                "A_is", "B_is", "candidate_system", "judge_verdict"]

    pairs_path = OUT_DIR / "human_eval_pairs.csv"
    key_path = OUT_DIR / "human_eval_key.csv"

    with open(pairs_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pair_cols)
        w.writeheader()
        for p in all_pairs:
            w.writerow({c: p.get(c, "") for c in pair_cols})

    with open(key_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=key_cols)
        w.writeheader()
        for k in all_keys:
            w.writerow({c: k.get(c, "") for c in key_cols})

    print(f"\nTotal items: {len(all_pairs)}")
    print(f"Blind sheet : {pairs_path}")
    print(f"Key (hidden): {key_path}")
    print("\nNext: each rater copies the blind sheet to "
          "human_eval_rater_<name>.csv, fills `choice` (A/B/tie), "
          "then run score_human_eval.py.")


if __name__ == "__main__":
    main()
