"""Seed TranslationMemory from aligned_master_chap_*.csv gold references.

The aligned CSVs use <MERGED_ABOVE> to denote rows whose translation is folded into
the previous non-merged reference. We reconstruct sentence-group pairs by
concatenating consecutive sources and pairing with the leading non-merged ref.

Usage:
    python seed_translation_memory.py \\
        --experiment qwen3-8b-local_yes \\
        --chapters 1 \\
        --source-dir evaluation/alignment_results_llama-3.1-8B \\
        [--max-pairs 50] [--score 5.0]

The seed is model-agnostic (uses `Source_Text` + `ref` columns, both shared across
alignment_results_* directories). `--source-dir` only picks which CSV to read.
"""

import argparse
import csv
import os
import sys

import const
from modules.translation_memory import TranslationMemory

MERGED = "<MERGED_ABOVE>"


def parse_sentence_groups(csv_path):
    """Yield (merged_source, merged_ref) tuples by collapsing <MERGED_ABOVE> rows.

    A new group starts at any row whose `ref` is not <MERGED_ABOVE>. All following
    <MERGED_ABOVE> rows extend its source side; the ref side stays as the leading
    cell. Empty refs are skipped (no anchor).
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_src = []
        current_ref = None
        for row in reader:
            src = (row.get("Source_Text") or "").strip()
            ref = (row.get("ref") or "").strip()
            if not src:
                continue
            if ref and ref != MERGED:
                if current_ref:
                    yield " ".join(current_src), current_ref
                current_src = [src]
                current_ref = ref
            else:
                if current_ref is None:
                    continue
                current_src.append(src)
        if current_ref:
            yield " ".join(current_src), current_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        required=True,
        help="experiment subdir under data/knowledge_base/, e.g. qwen3-8b-local_yes",
    )
    ap.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        default=[1],
        help="chapter numbers to seed from (default: just chap 1)",
    )
    ap.add_argument(
        "--source-dir",
        default="evaluation/alignment_results_llama-3.1-8B",
        help="alignment_results_* directory to read CSVs from",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=50,
        help="maximum pairs to seed (default 50)",
    )
    ap.add_argument(
        "--score",
        type=float,
        default=5.0,
        help="quality score to attach to seed entries (default 5.0, gold)",
    )
    args = ap.parse_args()

    target_dir = os.path.join(const.DATA_DIR, args.experiment)
    os.makedirs(target_dir, exist_ok=True)
    print(f"📂 Target TM dir: {target_dir}")

    items = []
    for chap in args.chapters:
        csv_path = os.path.join(
            args.source_dir, f"aligned_master_chap_{chap}.csv"
        )
        if not os.path.exists(csv_path):
            print(f"  ⚠️ Missing: {csv_path}")
            continue
        chap_pairs = list(parse_sentence_groups(csv_path))
        print(f"  chap {chap}: {len(chap_pairs)} sentence-groups parsed")
        items.extend(
            {"source": s, "translation": r, "score": args.score}
            for s, r in chap_pairs
        )
        if len(items) >= args.max_pairs:
            break

    items = items[: args.max_pairs]
    if not items:
        print("❌ No pairs to seed.")
        sys.exit(1)

    tm = TranslationMemory(target_dir)
    print(f"  pre-seed TM size: {len(tm)}")
    added = tm.add_batch(items)
    tm.save()
    print(f"✅ Added {added} new entries. Final TM size: {len(tm)}")

    sample = tm.retrieve(items[0]["source"], k=2)
    print("\n🔍 Sanity retrieval (query = first seed source):")
    for r in sample:
        print(f"  sim={r['similarity']:.3f}  EN: {r['source'][:60]}...")
        print(f"                          VI: {r['translation'][:60]}...")


if __name__ == "__main__":
    main()
