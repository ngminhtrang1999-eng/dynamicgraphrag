"""Split repaired LOTM Việt Hóa PDF text into web-novel chapters (v2).

Coverage established by LaBSE probe (_probe_pdf_coverage.py):
  Tập 1 body units ch01..ch08 cover EN 1-22; units ch13..ch20 cover EN 34-58;
  EN 23-33 are ABSENT from the PDF (units 09-12 missing).

v2 anchors each EN chapter's opening sentences ONLY within its mapped PDF
unit span (monotonic within unit), so absent chapters can't poison the
search. Last chapter of a unit ends at the unit end.

Usage (WSL graphrag env):  python pdf_split_chapters.py
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
REPAIRED = ROOT / "_pdf_repaired.txt"
EN_DIR = ROOT / "data" / "LOTM" / "chapters"
OUT_DIR = ROOT / "data" / "LOTM" / "chapters_vi_pdf_raw"

SENT_SPLIT = re.compile(r"(?<=[.!?…”])\s+|\n+")

# PDF body unit index (in document order, gap>5000 units) -> EN chapter range
UNIT_EN_RANGES = [
    (0, 1, 3), (1, 4, 6), (2, 7, 8), (3, 9, 11), (4, 12, 14), (5, 15, 16),
    (6, 17, 19), (7, 20, 22),
    (8, 34, 36), (9, 37, 39), (10, 40, 42), (11, 43, 45), (12, 46, 48),
    (13, 49, 51), (14, 52, 55), (15, 56, 58),
]


def en_chapter_text(n):
    for name in (f"chapter_{n:04d}.txt", f"chapter_{n}.txt"):
        p = EN_DIR / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    raise FileNotFoundError(f"EN chapter {n}")


def sentences(text, min_len=25):
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > min_len]


def main():
    full = REPAIRED.read_text(encoding="utf-8")
    chs = list(re.finditer(r"^[^\n]{0,20}(?:Chương|CHƯƠNG)\s*(\d+)[^\n]*$", full, re.M))
    units = []
    for i, m in enumerate(chs):
        nxt = chs[i + 1].start() if i + 1 < len(chs) else len(full)
        if nxt - m.start() > 5000:
            units.append((m.start(), nxt, m.group(0).strip()[:40]))
    print(f"body units: {len(units)}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for ui, en_a, en_b in UNIT_EN_RANGES:
        u_start, u_end, u_title = units[ui]
        block = full[u_start:u_end]
        vn_sents, vn_offsets, pos = [], [], 0
        for s in SENT_SPLIT.split(block):
            st = s.strip()
            idx = block.find(s, pos)
            if idx < 0:
                idx = pos
            pos = idx + len(s)
            if len(st) > 25:
                vn_sents.append(st)
                vn_offsets.append(idx)
        if not vn_sents:
            continue
        vn_emb = model.encode(vn_sents, batch_size=64, normalize_embeddings=True)

        bounds = {en_a: 0}  # first chapter starts at unit start
        scores = {en_a: None}
        search_from = 1
        for n in range(en_a + 1, en_b + 1):
            anchor_sents = sentences(en_chapter_text(n))[:3]
            if not anchor_sents:
                continue
            a_emb = model.encode(anchor_sents, batch_size=8, normalize_embeddings=True)
            sims = vn_emb @ a_emb.T
            window = sims[:, 0].copy()
            for j in range(1, sims.shape[1]):
                sh = np.full(len(vn_sents), 0.0)
                sh[: len(vn_sents) - j] = sims[j:, j]
                window += sh
            window[:search_from] = -1e9
            best = int(np.argmax(window))
            bounds[n] = best
            scores[n] = float(window[best]) / sims.shape[1]
            search_from = best + 3

        ordered = sorted(bounds.items())
        for k, (n, b) in enumerate(ordered):
            c0 = vn_offsets[b]
            c1 = vn_offsets[ordered[k + 1][1]] if k + 1 < len(ordered) else len(block)
            text = block[c0:c1].strip()
            text = re.sub(r"^.*Độ dài:.*?$", "", text, flags=re.M)
            text = re.sub(r"^.*Translator:.*?$", "", text, flags=re.M)
            text = re.sub(r"^[✫\s•]+$", "", text, flags=re.M)
            text = re.sub(r"\n{3,}", "\n\n", text)
            (OUT_DIR / f"chap_{n}.txt").write_text(text, encoding="utf-8")
            sc = scores.get(n)
            meta = {
                "chapter": n,
                "source": "LOTM - Việt Hóa.pdf (glyph-repaired, LaBSE-split v2)",
                "pdf_unit": u_title,
                "anchor_score": None if sc is None else round(sc, 3),
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "body_chars": len(text),
            }
            (OUT_DIR / f"chap_{n}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            report.append((n, len(text), sc))

    print("\nwritten:")
    for n, ln, sc in sorted(report):
        sstr = "unit-start" if sc is None else f"{sc:.3f}"
        flag = "  <-- CHECK" if (sc is not None and sc < 0.45) or ln < 4000 else ""
        print(f"  chap_{n}: {ln:>6} chars, score {sstr}{flag}")


if __name__ == "__main__":
    main()
