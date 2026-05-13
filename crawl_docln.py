"""Crawler for docln.net Vietnamese translation of Lord of the Mysteries.

We pull from the November 2022 web.archive.org snapshot (the live site has
been intermittently down / paywalled since then). The translation is by
"Kouji" — human-edited, much higher quality than the vntq auto-translation.

Coverage: Tập 01 (volume 1), chapters 3 through 20 — 18 chapters total. The
chapter IDs are not sequential (gaps from delisted/restructured posts), so
the URL list is hard-coded below from the TOC at archive snapshot
20221117071059.

NOTE on chapter numbering: docln chapter numbers do NOT 1:1 match the web
novel's English chapter_NNNN.txt files. docln uses the light-novel volume
edit (some web chapters got merged or re-cut). The crawl saves chapters
under docln's own numbering; manual content splitting is the user's job.

Output per chapter:
    data/LOTM/chapters_vi_docln_raw/chap_<N>.txt        # cleaned plain text
    data/LOTM/chapters_vi_docln_raw/chap_<N>.meta.json  # title, translator, etc.

Usage:
    python crawl_docln.py
    python crawl_docln.py --only 11        # fetch a single chapter
    python crawl_docln.py --overwrite      # re-fetch existing
"""

import argparse
import html as html_mod
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ARCHIVE_PREFIX = "https://web.archive.org/web/20221117071059/"
DOCLN_BASE = "https://docln.net/truyen/3151-the-lord-of-the-mysteries/"
DEFAULT_OUT = Path("data/LOTM/chapters_vi_docln_raw")

# Tập 01 chapter URL map: docln Chương N -> slug.
# c79298/c79299 have a slug typo (both "chuong-03") but c79299 is actually
# Chương 04 - Nhật Ký Của Klein per the page title; we hit the URL by ID.
TAP_01_CHAPTERS = {
    3:  "c79298-chuong-03-buoi-tu-hop-tarot",
    4:  "c79299-chuong-03-buoi-tu-hop-tarot",        # slug typo on docln; content is Chương 04
    5:  "c79303-chuong-05",
    6:  "c79304-chuong-06-loi-moi-tu-dunn",
    7:  "c79305-chuong-07-bo-phan-hanh-dong-dac-biet",
    8:  "c79306-chuong-08",
    9:  "c79307-chuong-09-ke-nhin-trom-tu-goc-toi",
    10: "c79309-chuong-10-luyen-tap-ban-sung",
    11: "c79310-chuong-11-ma-duoc-nha-boi-toan",
    12: "c79311-chuong-12-cong-tac-linh-thi",
    13: "c79312-chuong-13-loi-nhac-nho-tu-ke-kho",
    14: "c79313-chuong-14-cau-lac-bo-boi-toan",
    15: "c79318-chuong-15-khoa-trinh-than-bi-hoc",
    16: "c79319-chuong-16-nhiem-vu",
    17: "c79320-chuong-17-chan-dung-ray-bieber",
    18: "c79321-chuong-18-nghi-le-ma-thuat-dac-biet",
    19: "c79322-chuong-19-khach-hang-dau-tien",
    20: "c79432-chuong-20-sap-xep-va-tong-ket",
}


def parse_args():
    p = argparse.ArgumentParser(description="Crawl LOTM Vietnamese chapters from docln.net (via web.archive.org)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sleep", type=float, default=2.5, help="Polite delay between archive requests (default 2.5)")
    p.add_argument("--only", type=int, help="Fetch only this chapter number")
    p.add_argument("--overwrite", action="store_true", help="Re-fetch even if output exists")
    return p.parse_args()


_chapter_content_re = re.compile(
    r'<div[^>]*id="chapter-content"[^>]*>(.*?)</div>\s*<div[^>]*class="ln-comment',
    re.DOTALL,
)
_title_re = re.compile(r'<h4[^>]*class="title-item"[^>]*>\s*(.*?)\s*</h4>', re.DOTALL)
_translator_re = re.compile(r"Translator:\s*([^\n<]+)")
_close_p_re = re.compile(r"</p>", flags=re.IGNORECASE)
_tag_re = re.compile(r"<[^>]+>")
_multi_ws_re = re.compile(r"[ \t]+")
_multi_nl_re = re.compile(r"\n{3,}")


def extract_title(page_html: str) -> str:
    m = _title_re.search(page_html)
    if not m:
        return ""
    text = _tag_re.sub("", m.group(1))
    return html_mod.unescape(text.strip())


def extract_body(page_html: str):
    """Return (body_text, translator_name_or_empty)."""
    m = _chapter_content_re.search(page_html)
    if not m:
        # Fallback selector — chapter-content div without trailing comment block
        m = re.search(r'<div[^>]*id="chapter-content"[^>]*>(.*?)</div>\s*</section>', page_html, re.DOTALL)
    if not m:
        raise RuntimeError("Could not locate #chapter-content div")
    body_html = m.group(1)

    # Pull translator credit out of body before paragraph break processing
    tm = _translator_re.search(body_html)
    translator = tm.group(1).strip() if tm else ""

    # Each </p> becomes a paragraph break
    text = _close_p_re.sub("\n\n", body_html)
    # Strip remaining tags
    text = _tag_re.sub("", text)
    text = html_mod.unescape(text)
    text = _multi_ws_re.sub(" ", text)

    # Drop translator credit line + "✫ ✫ ✫" separator that always opens
    lines = [ln.strip() for ln in text.split("\n")]
    cleaned = []
    for ln in lines:
        if not ln:
            cleaned.append("")
            continue
        if ln.startswith("Translator:"):
            continue
        # Drop simple decorative star lines
        if re.fullmatch(r"[✫✦✧✩✬✭✮✯★☆\s]+", ln):
            continue
        cleaned.append(ln)
    text = "\n".join(cleaned)
    text = _multi_nl_re.sub("\n\n", text)
    text = text.strip("\n")
    return text, translator


def fetch(sess, url: str) -> str:
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def fetch_chapter(sess, chap_num: int, slug: str):
    url = ARCHIVE_PREFIX + DOCLN_BASE + slug
    html = fetch(sess, url)
    title = extract_title(html)
    body_text, translator = extract_body(html)
    return {
        "chapter": chap_num,
        "slug": slug,
        "url": url,
        "title": title,
        "translator": translator,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "body_bytes": len(body_text.encode("utf-8")),
        "body": body_text,
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; LOTM-ResearchCrawler)"})

    targets = list(TAP_01_CHAPTERS.items())
    if args.only is not None:
        targets = [(args.only, TAP_01_CHAPTERS[args.only])]

    print(f"Fetching {len(targets)} chapter(s) from docln (via web.archive.org)")
    print(f"Output dir: {args.out_dir}")

    successes, failures = 0, []
    for chap_num, slug in targets:
        body_path = args.out_dir / f"chap_{chap_num}.txt"
        meta_path = args.out_dir / f"chap_{chap_num}.meta.json"
        if body_path.exists() and meta_path.exists() and not args.overwrite:
            print(f"  [skip] chap {chap_num}: already crawled")
            continue
        try:
            ch = fetch_chapter(sess, chap_num, slug)
        except Exception as e:
            print(f"  [FAIL] chap {chap_num} ({slug}): {e}")
            failures.append(chap_num)
            continue

        body_path.write_text(ch["body"], encoding="utf-8")
        meta = {k: v for k, v in ch.items() if k != "body"}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [ok]   chap {chap_num:>2}  {ch['body_bytes']:>6}B  title={ch['title']!r}  translator={ch['translator']!r}")
        successes += 1
        if chap_num != targets[-1][0]:
            time.sleep(args.sleep)

    print()
    print(f"Done: {successes} fetched, {len(failures)} failed")
    if failures:
        print(f"  failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
