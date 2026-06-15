"""Extract LOTM Vietnamese fan-translation chapters from the 'LOTM - Việt Hóa' PDF.

The PDF is organised as multiple booklets ("Tập 01".."Tập 07"), each restarting
chapter numbering at 1. Global chapter number = cumulative count across booklets.
Text is pulled with PyMuPDF (plain reading order, no table detection artifacts),
NFC-normalised, and written one file per global chapter to
data/LOTM/chapters_vi_pdf_raw/chap_N.txt (+ chap_N.meta.json), matching the
naming convention of the other chapters_vi_* raw dirs.

Usage: python extract_pdf_chapters.py [--pdf PATH] [--out DIR] [--max-chap N]
"""

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

DEFAULT_PDF = r"C:\Users\trang\Downloads\LOTM - Việt Hóa.pdf"
DEFAULT_OUT = Path(__file__).parent / "data" / "LOTM" / "chapters_vi_pdf_raw"

TAP_RE = re.compile(r"^\s*Tập\s*(\d+)\b[^\n]*$", re.MULTILINE)
CHAP_RE = re.compile(r"^\s*Chương\s*(\d+)\s*[:.]?\s*([^\n]*)$", re.MULTILINE)


def extract_full_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        text = unicodedata.normalize("NFC", text)
        pages.append(text)
    return "\n".join(pages)


def clean_body(text: str) -> str:
    # drop page-number-only lines and decorative separators
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.fullmatch(r"[-—_=*~.•]{3,}", s):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    # collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-chap", type=int, default=None,
                    help="stop after this global chapter number")
    args = ap.parse_args()

    full = extract_full_text(args.pdf)

    # locate booklet starts; skip the table of contents by keeping only the
    # LAST run of Tập markers ordering (body comes after TOC duplicates)
    tap_positions = [(m.start(), int(m.group(1))) for m in TAP_RE.finditer(full)]
    # body Tập markers: the final monotone-increasing run 1..7
    body_taps = []
    for pos, num in tap_positions:
        while body_taps and body_taps[-1][1] >= num:
            body_taps.pop()
        body_taps.append((pos, num))
    if not body_taps:
        raise SystemExit("no Tập markers found")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_no = 0
    written = []
    for ti, (tap_pos, tap_num) in enumerate(body_taps):
        tap_end = body_taps[ti + 1][0] if ti + 1 < len(body_taps) else len(full)
        block = full[tap_pos:tap_end]
        chaps = list(CHAP_RE.finditer(block))
        # keep only real chapter bodies (TOC remnants have tiny gaps)
        starts = []
        for ci, m in enumerate(chaps):
            nxt = chaps[ci + 1].start() if ci + 1 < len(chaps) else len(block)
            if nxt - m.start() > 800:
                starts.append((m.start(), nxt, int(m.group(1)), m.group(2).strip()))
        for s, e, local_no, title in starts:
            global_no += 1
            if args.max_chap and global_no > args.max_chap:
                break
            body = clean_body(block[s:e])
            path = out_dir / f"chap_{global_no}.txt"
            path.write_text(body, encoding="utf-8")
            meta = {
                "chapter": global_no,
                "tap": tap_num,
                "local_chapter": local_no,
                "title": title,
                "source": "LOTM - Việt Hóa.pdf",
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "body_chars": len(body),
            }
            (out_dir / f"chap_{global_no}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append((global_no, tap_num, local_no, title, len(body)))

    print(f"wrote {len(written)} chapters to {out_dir}")
    for g, t, l, title, n in written:
        print(f"chap_{g:>2}  (Tập {t:02d} / Chương {l:02d})  {n:>6} chars  {title[:50]}")


if __name__ == "__main__":
    main()
