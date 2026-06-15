"""Map each PDF body 'Chương' unit to its best-matching EN web chapter (LaBSE).

Decides definitively which EN chapters the LOTM Việt Hóa PDF actually covers,
and where the gaps are.
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
REPAIRED = ROOT / "_pdf_repaired.txt"
EN_DIR = ROOT / "data" / "LOTM" / "chapters"

SENT_SPLIT = re.compile(r"(?<=[.!?…”])\s+|\n+")


def sents(text, n=5, skip=1):
    out = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 30]
    return out[skip: skip + n]


full = REPAIRED.read_text(encoding="utf-8")
chs = list(re.finditer(r"^[^\n]{0,20}(?:Chương|CHƯƠNG)\s*(\d+)[^\n]*$", full, re.M))
units = []
for i, m in enumerate(chs):
    nxt = chs[i + 1].start() if i + 1 < len(chs) else len(full)
    if nxt - m.start() > 5000:
        units.append((m.group(0).strip()[:48], m.start(), nxt))
print(f"PDF body units: {len(units)}")

en_chaps = {}
for n in range(1, 76):
    for name in (f"chapter_{n:04d}.txt", f"chapter_{n}.txt"):
        p = EN_DIR / name
        if p.exists():
            en_chaps[n] = sents(p.read_text(encoding="utf-8", errors="ignore"))
            break

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")

vn_texts, vn_meta = [], []
for title, s, e in units:
    for sent in sents(full[s:e], n=5, skip=2):
        vn_texts.append(sent)
        vn_meta.append(title)
en_texts, en_meta = [], []
for n, ss in en_chaps.items():
    for sent in ss:
        en_texts.append(sent)
        en_meta.append(n)

vn_emb = model.encode(vn_texts, batch_size=64, normalize_embeddings=True)
en_emb = model.encode(en_texts, batch_size=64, normalize_embeddings=True)
sim = vn_emb @ en_emb.T

en_meta = np.array(en_meta)
print(f"{'PDF unit':<50} {'best EN':>8} {'votes':>14} {'top sim':>8}")
for title, s, e in units:
    rows = [i for i, t in enumerate(vn_meta) if t == title]
    if not rows:
        continue
    best_per_sent = [(int(en_meta[int(np.argmax(sim[r]))]), float(sim[r].max())) for r in rows]
    votes = {}
    for c, sc in best_per_sent:
        votes[c] = votes.get(c, 0) + 1
    top = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
    avg_top = max(sc for _, sc in best_per_sent)
    print(f"{title:<50} {top[0][0]:>8} {str(top):>14} {avg_top:>8.3f}")
