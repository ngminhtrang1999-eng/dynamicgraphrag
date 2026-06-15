"""Repair the LOTM Việt Hóa PDF text: Ethiopic tone-glyph decoding + split-syllable rejoin.

The PDF body font maps Vietnamese composite diacritics (circumflex/breve +
tone) to Ethiopic codepoints, and the extractor inserts spurious spaces inside
syllables. This module repairs both, scoring candidates against a syllable
frequency table built from the project's clean Vietnamese corpus
(chapters_vi/, translations/reference/, google_translate/).

Usage:
    python pdf_text_repair.py            # repair full PDF text -> _pdf_repaired.txt + stats
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PDF_CACHE = ROOT / "_pymupdf_full.txt"
OUT_TXT = ROOT / "_pdf_repaired.txt"

# glyph -> candidate composite vowels (lowercase). The base letter preceding
# the glyph selects which candidates are type-compatible.
GLYPH_CANDIDATES = {
    "ቷ": ["ấ", "ế", "ố", "ắ"],
    "ቹ": ["ầ", "ề", "ồ"],
    "ቻ": ["ẩ", "ể", "ổ"],
    "ች": ["ẫ", "ễ", "ỗ"],
    "ቿ": ["ắ", "ậ", "ệ", "ộ", "ặ"],
    "ኁ": ["ằ", "ầ"],
    "ኃ": ["ẳ", "ẩ"],
    "ኅ": ["ẵ", "ẫ"],
    "ኀ": ["ậ", "ặ"],
    # uppercase variants
    "ቶ": ["Ờ", "Ấ", "Ầ"],
    "ቸ": ["Ấ", "Ầ", "Ố"],
    "ቺ": ["Ẩ", "Ấ", "Ổ"],
    "ቼ": ["Ấ", "Ẩ", "Ẫ"],
    "ቾ": ["Ế", "Ề"],
}
# the glyph REPLACES the diacritics of the PRECEDING vowel letter:
# "đe<ቷ>n" -> "đến" (e+ቷ -> ế). Base letter must match candidate's base.
BASE_OF = {}
for _g, _cands in GLYPH_CANDIDATES.items():
    for _c in _cands:
        BASE_OF[_c] = unicodedata.normalize("NFD", _c)[0].lower()

ETH_RE = re.compile(r"[ሀ-፿]")
WORD_RE = re.compile(r"[\wሀ-፿]+", re.UNICODE)

VI_LOWER = "aăâbcdđeêghiklmnoôơpqrstuưvxy" \
           "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"


def build_syllable_freq():
    """Syllable + bigram frequency from the project's clean VN text."""
    freq = Counter()
    bigrams = Counter()
    sources = (
        list((ROOT / "data/LOTM/chapters_vi").glob("chap_*.txt"))
        + list((ROOT / "data/LOTM/translations/reference").glob("ref_*.txt"))
        + list((ROOT / "data/LOTM/translations/google_translate").glob("translated_*.txt"))
    )
    for p in sources:
        text = unicodedata.normalize("NFC", p.read_text(encoding="utf-8", errors="ignore"))
        words = [w.lower() for w in re.findall(r"[a-zA-Zà-ỹÀ-ỸđĐ]+", text)]
        freq.update(words)
        bigrams.update(zip(words, words[1:]))
    return freq, bigrams


def candidates_for_word(word):
    """Yield repaired variants of a word containing Ethiopic glyphs."""
    spots = [i for i, c in enumerate(word) if ETH_RE.match(c)]
    if not spots:
        return [word]
    variants = [word]
    for i in reversed(spots):  # right-to-left keeps indices stable
        glyph = word[i]
        new_variants = []
        for v in variants:
            prev = v[i - 1] if i > 0 else ""
            for cand in GLYPH_CANDIDATES.get(glyph, []):
                if prev.lower() == BASE_OF.get(cand, "?"):
                    # replace base letter + glyph with composite (case of base)
                    comp = cand if prev.islower() or not prev.isalpha() else cand.upper()
                    if prev.isupper():
                        comp = cand.upper()
                    new_variants.append(v[: i - 1] + comp + v[i + 1:])
            # glyph with no compatible base: drop glyph as last resort
            if not new_variants:
                new_variants.append(v[:i] + v[i + 1:])
        variants = new_variants
    return variants


def repair_glyphs(text, freq, bigrams):
    stats = {"words": 0, "resolved": 0, "ambiguous": 0, "unresolved": 0}
    ambiguous_log = Counter()

    # process token stream so bigram context is available
    tokens = []  # (is_word, text)
    pos = 0
    for m in WORD_RE.finditer(text):
        if m.start() > pos:
            tokens.append((False, text[pos:m.start()]))
        tokens.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        tokens.append((False, text[pos:]))

    word_idx = [i for i, (isw, _) in enumerate(tokens) if isw]

    widx_pos = {wi: j for j, wi in enumerate(word_idx)}

    def neighbor(wi, offset):
        k = widx_pos[wi] + offset
        if 0 <= k < len(word_idx):
            return tokens[word_idx[k]][1].lower()
        return ""

    for wi in word_idx:
        w = tokens[wi][1]
        if not ETH_RE.search(w):
            continue
        stats["words"] += 1
        cands = candidates_for_word(w)
        prev_w, next_w = neighbor(wi, -1), neighbor(wi, +1)

        def score(c):
            lw = c.lower()
            uni = freq.get(lw, 0)
            big = bigrams.get((prev_w, lw), 0) + bigrams.get((lw, next_w), 0)
            return (big, uni)

        scored = sorted(cands, key=score, reverse=True)
        best = scored[0]
        b_big, b_uni = score(best)
        if b_uni == 0 and b_big == 0:
            stats["unresolved"] += 1
            ambiguous_log[w] += 1
        elif len(scored) > 1:
            s_big, s_uni = score(scored[1])
            if s_big == b_big and s_uni >= max(1, b_uni // 3):
                stats["ambiguous"] += 1
                ambiguous_log[f"{w} ({prev_w}_) -> {best}|{scored[1]}"] += 1
            stats["resolved"] += 1
        else:
            stats["resolved"] += 1
        tokens[wi] = (True, best)

    out = "".join(t for _, t in tokens)
    return out, stats, ambiguous_log


def rejoin_split_syllables(text, freq):
    """Fix extractor-inserted spaces inside syllables: 'Ngườ i' -> 'Người'."""
    joined = {"n": 0}
    # fragment + space + short tail (consonants or i/u/y/n/m/ng/nh/ch/c/t/p/o/a/e/u...)
    pat = re.compile(r"([A-ZĐa-zđàáảãạ-ỹÀ-Ỹ]*[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ])[ ]([a-zđ]{1,3})\b")

    def try_join(m):
        head, tail = m.group(1), m.group(2)
        merged = head + tail
        last_word = re.findall(r"[\wàáảãạ-ỹĐđ]+$", merged)
        if not last_word:
            return m.group(0)
        mw = last_word[0].lower()
        # join only when merged final word is a known syllable AND the tail
        # alone is not a common standalone word
        if freq.get(mw, 0) >= 3 and freq.get(tail.lower(), 0) < 50:
            joined["n"] += 1
            return merged
        return m.group(0)

    prev = None
    while prev != text:  # iterate: 'Ngư ờ i' style double splits
        prev = text
        text = pat.sub(try_join, text)
    return text, joined["n"]


def main():
    freq, bigrams = build_syllable_freq()
    print(f"syllable table: {len(freq)} distinct, {len(bigrams)} bigrams")
    full = PDF_CACHE.read_text(encoding="utf-8")
    repaired, stats, amb = repair_glyphs(full, freq, bigrams)
    repaired, n_joined = rejoin_split_syllables(repaired, freq)
    OUT_TXT.write_text(repaired, encoding="utf-8")
    print(f"glyph words: {stats}")
    print(f"split syllables rejoined: {n_joined}")
    print(f"residual ethiopic chars: {len(ETH_RE.findall(repaired))}")
    print("--- top ambiguous/unresolved ---")
    for k, n in amb.most_common(15):
        print(f"  {n:>5}  {k}")


if __name__ == "__main__":
    main()
