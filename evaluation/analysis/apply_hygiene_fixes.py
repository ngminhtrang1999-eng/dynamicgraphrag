"""Hygiene-1 enforcement: clone the TM experiment dir and apply name-policy
corrections to glossary + TM translations.

Idempotent: re-running overwrites the clone from scratch each time.

Output: data/knowledge_base/qwen3-8b-local_yes_hygiene/  (cloned + fixed)

Corrections applied (Track 1 findings, see qualitative_gap_analysis.md):
- Glossary corrections for known mistranslations.
- TM translation rewrites: Zhou Mingrui → Chu Minh Thụy.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "data" / "knowledge_base" / "qwen3-8b-local_yes_tm"
DST_DIR = ROOT / "data" / "knowledge_base" / "qwen3-8b-local_yes_hygiene"

# Glossary corrections: replace or add. Format: english_term -> (translation, type)
GLOSSARY_OVERRIDES = {
    "Mr. Fool": ("Kẻ Khờ", "TITLE"),
    "The Fool": ("Kẻ Khờ", "TITLE"),
    "Fool": ("Kẻ Khờ", "TITLE"),
    "Beyonder": ("phi phàm", "CONCEPT"),
    "Beyonders": ("phi phàm", "CONCEPT"),
    "gray fog": ("sương mù xám", "LOCATION"),
    "Gray Fog": ("sương mù xám", "LOCATION"),
    "Grayish-white fog": ("sương mù xám trắng", "LOCATION"),
    "Nighthawks": ("Kẻ Gác Đêm", "ORG"),
    "Nighthawk": ("Kẻ Gác Đêm", "ORG"),
    # Add PERSON entries (missing entirely from current glossary)
    "Zhou Mingrui": ("Chu Minh Thụy", "PERSON"),
    "Klein": ("Klein", "PERSON"),
    "Klein Moretti": ("Klein Moretti", "PERSON"),
    "Audrey Hall": ("Audrey Hall", "PERSON"),
    "Audrey": ("Audrey", "PERSON"),
    "Alger Wilson": ("Alger Wilson", "PERSON"),
    "Alger": ("Alger", "PERSON"),
    "Welch": ("Welch", "PERSON"),
    "Welch McGovern": ("Welch McGovern", "PERSON"),
    "Benson": ("Benson", "PERSON"),
    "Benson Moretti": ("Benson Moretti", "PERSON"),
    "Melissa": ("Melissa", "PERSON"),
    "Melissa Moretti": ("Melissa Moretti", "PERSON"),
    "Mountbatten": ("Mountbatten", "PERSON"),
    "Bitsch Mountbatten": ("Bitsch Mountbatten", "PERSON"),
    "Dunn Smith": ("Dunn Smith", "PERSON"),
    "Dunn": ("Dunn", "PERSON"),
}

# TM-level rewrites in the `translation` field.
# Format: substring_to_find -> replacement.
# Applied ONLY when the source side contains the corresponding english_form.
TM_REWRITES = [
    {
        "source_must_contain": "Zhou Mingrui",
        "translation_replace": [("Zhou Mingrui", "Chu Minh Thụy")],
    },
]


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: source experiment dir not found: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    # 1. Fresh clone (idempotent).
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    shutil.copytree(SRC_DIR, DST_DIR)
    print(f"✓ Cloned {SRC_DIR.name} → {DST_DIR.name}")

    # 2. Glossary fixes.
    glossary_path = DST_DIR / "term_glossary.json"
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    changed, added = 0, 0
    for term, (translation, type_) in GLOSSARY_OVERRIDES.items():
        existing = glossary.get(term)
        if existing is None:
            glossary[term] = {"translation": translation, "type": type_}
            added += 1
        else:
            old_t = existing.get("translation") if isinstance(existing, dict) else existing
            if old_t != translation:
                glossary[term] = {"translation": translation, "type": type_}
                changed += 1
                print(f"  glossary fix: {term!r}: {old_t!r} → {translation!r}")
    glossary_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Glossary: {changed} changed, {added} added (total {len(glossary)} entries)")

    # 3. TM translation rewrites.
    tm_path = DST_DIR / "translation_memory.json"
    tm = json.loads(tm_path.read_text(encoding="utf-8"))
    n_rewritten = 0
    for rule in TM_REWRITES:
        src_substr = rule["source_must_contain"]
        for entry in tm:
            if src_substr not in entry["source"]:
                continue
            orig = entry["translation"]
            new = orig
            for find, replace in rule["translation_replace"]:
                new = new.replace(find, replace)
            if new != orig:
                entry["translation"] = new
                n_rewritten += 1
    tm_path.write_text(json.dumps(tm, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ TM: {n_rewritten} translation entries rewritten ({len(tm)} total)")

    # 4. Reset translation_progress for chap 6-10 so the hygiene re-run starts
    # fresh on each chapter (the source clone has all chap 6-10 marked done,
    # which would cause the resume logic to skip every chunk). Leave
    # global_tracking.last_processed_chap unchanged so Phase-1 mining stays
    # skipped — we don't want mining to overwrite the corrected glossary.
    prog_path = DST_DIR / "translation_progress.json"
    progress = json.loads(prog_path.read_text(encoding="utf-8"))
    cleared = []
    for chap in [6, 7, 8, 9, 10]:
        key = str(chap)
        if key in progress:
            del progress[key]
            cleared.append(chap)
    prog_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Progress: cleared chap {cleared} (forces fresh re-translation)")

    print(f"\nDone. Hygiene experiment dir ready: {DST_DIR}")


if __name__ == "__main__":
    main()
