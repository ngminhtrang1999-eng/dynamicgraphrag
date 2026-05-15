"""Hygiene-2: output contamination detector for Vietnamese translation output.

Catches the two failure modes that the qualitative analysis surfaced as
high-leverage and not solvable by retrieval changes (see
qualitative_gap_analysis.md, modes "CJK / English token leak" and "Name
romanization inconsistency"):

  1. CJK characters mixed mid-Vietnamese-word ("kinh悚")
  2. Pinyin name leak when policy says use Hán-Việt ("Zhou Mingrui" →
     should be "Chu Minh Thụy")
  3. Stray English tokens mid-sentence ("ripples", "Whoosh")

Used by core/pipeline.py PostEditingPipeline to decide between the draft and
the post-edit output (preferring whichever is cleaner) and to surface a
diagnostic.
"""
from __future__ import annotations
import re
from typing import Iterable

# CJK Unified Ideographs + Extension A + Compatibility Ideographs.
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")

# Pinyin / English forms that the name policy requires to be re-rendered.
# Surface forms only; we don't tokenize. Values are the canonical Hán-Việt /
# kept form for the diagnostic message — detection doesn't use them.
_PINYIN_LEAK = {
    "Zhou Mingrui": "Chu Minh Thụy",
}

# A bare lowercase-ASCII regex over-flags Vietnamese words without diacritics
# (anh, trong, vang, ban, etc.). Use a curated denylist of English content
# words observed leaking in the qualitative analysis instead — high precision,
# low recall is the right trade-off for an automated filter.
_LEAK_DENYLIST = {
    "ripples", "ripple",
    "whoosh", "swoosh",
    "trademark",
    "Whoosh", "Swoosh",  # capitalized onomatopoeia leak
}
_LEAK_DENYLIST_RE = re.compile(
    r"\b(" + "|".join(sorted(_LEAK_DENYLIST, key=len, reverse=True)) + r")\b"
)


def detect_cjk(text: str) -> list[str]:
    return _CJK_RE.findall(text)


def detect_pinyin_leak(text: str) -> list[str]:
    return [p for p in _PINYIN_LEAK if p in text]


def detect_english_tokens(text: str) -> list[str]:
    return _LEAK_DENYLIST_RE.findall(text)


def contamination_score(text: str) -> tuple[int, dict]:
    """Returns (score, breakdown). Higher score = more contaminated.

    Score weights match severity from the qualitative analysis:
    - CJK character: weight 5 (red-flag, e.g. "kinh悚")
    - Pinyin leak: weight 3 (clearly wrong name convention)
    - English token: weight 1 (each occurrence; noisy)
    """
    cjk = detect_cjk(text)
    pinyin = detect_pinyin_leak(text)
    english = detect_english_tokens(text)
    score = 5 * len(cjk) + 3 * len(pinyin) + 1 * len(english)
    return score, {"cjk": cjk, "pinyin": pinyin, "english": english}


def is_contaminated(text: str) -> bool:
    score, _ = contamination_score(text)
    return score > 0


def pick_cleaner(*candidates: str) -> tuple[int, str]:
    """Given several candidate translations, returns (winning_index, text)
    of the cleanest one. Ties broken by argument order (so caller's preferred
    candidate wins ties)."""
    best_idx, best_score, best_text = 0, None, candidates[0]
    for i, t in enumerate(candidates):
        s, _ = contamination_score(t)
        if best_score is None or s < best_score:
            best_idx, best_score, best_text = i, s, t
    return best_idx, best_text


__all__ = [
    "detect_cjk",
    "detect_pinyin_leak",
    "detect_english_tokens",
    "contamination_score",
    "is_contaminated",
    "pick_cleaner",
]
