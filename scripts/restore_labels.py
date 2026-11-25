"""Build or restore the published ID+label copies of the LLM-judge verdicts.

The public repository ships only identifiers and labels. Text columns (source,
reference, candidate translations, judge rationales, free-text notes) are removed
because the novel and its translations are copyrighted.

    python scripts/restore_labels.py           # labels/llm_judge -> evaluation/llm_judge (no overwrite)
    python scripts/restore_labels.py --build   # evaluation/llm_judge -> labels/llm_judge (maintainer only)

The statistics scripts (evaluation/paper_stats.py, evaluation/revision_stats.py,
analyze_new_judgments.py, analyze_position_swap.py, ...) only read label columns,
so they run unchanged on the restored files. Scripts that mine rationales or
quote sentences need the full outputs, which you regenerate yourself.
"""
import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "evaluation" / "llm_judge"
PUBLIC = ROOT / "labels" / "llm_judge"

ID_COLS = {"chapter", "sentence_id", "pair_id", "chap", "sid", "set_name", "item_id"}
LABEL_COLS = {
    "call_1_choice", "call_2_choice_unswapped", "final_choice", "judge_model",
    "candidate_col", "accuracy", "fluency", "register", "cultural", "style",
    "A_system", "B_system", "A_is", "B_is", "verdict", "winner", "choice", "your_choice",
}
KEEP = ID_COLS | LABEL_COLS


def build():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for src in sorted(FULL.glob("*.csv")):
        if src.name.startswith("_"):  # smoke tests / quota probes
            continue
        with src.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = [c for c in reader.fieldnames if c in KEEP]
            rows = [{c: r[c] for c in cols} for r in reader]
        if not set(cols) - ID_COLS:  # nothing but identifiers left (input sheets)
            skipped += 1
            continue
        with (PUBLIC / src.name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        written += 1
    print(f"wrote {written} label files to {PUBLIC.relative_to(ROOT)}, skipped {skipped} text-only sheets")


def restore():
    FULL.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(PUBLIC.glob("*.csv")):
        dst = FULL / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"copied {copied} files into {FULL.relative_to(ROOT)} (existing files left untouched)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="maintainer: regenerate labels/ from full outputs")
    build() if ap.parse_args().build else restore()
