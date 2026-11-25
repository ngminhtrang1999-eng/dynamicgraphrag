"""Inventory the broken Ethiopic-range glyphs in the LOTM Việt Hóa PDF body text.

The body font maps Vietnamese tone-composite glyphs to Ethiopic codepoints.
For each distinct Ethiopic char, show frequency and most common (prev, next)
contexts so the diacritic-repair mapping can be derived.
"""
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import fitz

PDF = os.environ.get("LOTM_VI_PDF", "")  # path to your own copy of the PDF
CACHE = Path(__file__).parent / "_pymupdf_full.txt"

if CACHE.exists():
    full = CACHE.read_text(encoding="utf-8")
else:
    doc = fitz.open(PDF)
    full = "\n".join(unicodedata.normalize("NFC", p.get_text("text")) for p in doc)
    CACHE.write_text(full, encoding="utf-8")

eth_chars = Counter(c for c in full if "ሀ" <= c <= "፿")
print(f"distinct ethiopic glyphs: {len(eth_chars)}, total occurrences: {sum(eth_chars.values())}")

contexts = defaultdict(Counter)
for m in re.finditer(r"(\w{0,4})([ሀ-፿])(\w{0,4})", full):
    contexts[m.group(2)][(m.group(1)[-2:], m.group(3)[:2])] += 1

for ch, cnt in eth_chars.most_common():
    top = contexts[ch].most_common(6)
    ctx_str = "  ".join(f"{a}_{b}×{n}" for (a, b), n in top)
    print(f"U+{ord(ch):04X} {ch} n={cnt:>6}  {ctx_str}")
