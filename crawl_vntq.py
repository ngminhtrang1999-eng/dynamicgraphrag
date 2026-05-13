"""Crawler for vietnamthuquan.eu — fetches a range of LOTM chapters in Vietnamese.

This is the Vietnamese translation of *Quỷ Bí Chi Chủ* (= Lord of the Mysteries),
hosted as a Vietnamese fan/auto-translation. We use it as a SEED reference for
the Translation Memory and as a LoRA hold-out source. Quality is uneven (the
site is partially auto-translated), so we crawl first and then the user edits
via `vntq_edit_app.py`.

Site mechanics:
  - Novel ID: tuaid=31338  (URL token: 2qtqv3m3237nnnvnnnnn4n31n343tq83a3q3m3237nvn)
  - Chapters fetched via POST /truyen/chuonghoi_moi.aspx
      body: tuaid=31338&chuongid=<N>
  - Mapping: chuongid = chapter_number + 1 (chuongid=1 is preface)
  - Response is 4 sections separated by `--!!tach_noi_dung!!--`:
      section 0: full <html><head>...</head><body>...
      section 1: title + prev/next nav (top)
      section 2: chapter body wrapped in <p>...</p>      <-- WHAT WE WANT
      section 3: prev/next nav (bottom)

Output per chapter:
    data/LOTM/chapters_vi_raw/chap_<N>.txt        # cleaned plain text
    data/LOTM/chapters_vi_raw/chap_<N>.meta.json  # {chapter, chuongid, title, fetched_at, body_bytes}

Usage:
    python crawl_vntq.py --start 11 --end 25
"""

import argparse
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "http://vietnamthuquan.eu/truyen/"
TID = "2qtqv3m3237nnnvnnnnn4n31n343tq83a3q3m3237nvn"
NOVEL_ID = "31338"
SEPARATOR = "--!!tach_noi_dung!!--"
DEFAULT_OUT = Path("data/LOTM/chapters_vi_raw")


def parse_args():
    p = argparse.ArgumentParser(description="Crawl LOTM Vietnamese chapters from vietnamthuquan.eu")
    p.add_argument("--start", type=int, required=True, help="First chapter number (inclusive)")
    p.add_argument("--end", type=int, required=True, help="Last chapter number (inclusive)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT})")
    p.add_argument("--sleep", type=float, default=2.0, help="Polite delay between requests in seconds (default 2)")
    p.add_argument("--overwrite", action="store_true", help="Re-fetch even if output file exists")
    return p.parse_args()


def make_session():
    """Prime cookies by hitting the TOC page once."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; LOTM-ResearchCrawler)"})
    sess.get(BASE + f"truyen.aspx?tid={TID}&cochu=", timeout=30)
    return sess


def fetch_chuongid(sess, cid: int) -> str:
    r = sess.post(
        BASE + "chuonghoi_moi.aspx",
        data={"tuaid": NOVEL_ID, "chuongid": str(cid)},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    return r.text


_p_open_re = re.compile(r"<p[^>]*>", flags=re.IGNORECASE)
_close_re = re.compile(r"</p>", flags=re.IGNORECASE)
_br_re = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_tag_re = re.compile(r"<[^>]+>")
_multi_ws_re = re.compile(r"[ \t]+")
_multi_nl_re = re.compile(r"\n{3,}")


def extract_chapter_text(html_body: str) -> str:
    """Strip HTML, decode entities, normalize whitespace.

    The site wraps each paragraph in <p>...</p>; we render those as blank-line-
    separated paragraphs. <br> becomes a newline. Other tags are stripped.
    """
    text = html_body
    text = _p_open_re.sub("", text)
    text = _close_re.sub("\n\n", text)
    text = _br_re.sub("\n", text)
    text = _tag_re.sub("", text)
    text = html.unescape(text)
    text = _multi_ws_re.sub(" ", text)
    # Strip each line on both sides — the source wraps <p>\n  body \n</p>,
    # leaving a leading space on every line after tag removal. Empty lines
    # collapse with the multi-newline rule below.
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _multi_nl_re.sub("\n\n", text)
    while text.startswith("\n"):
        text = text[1:]
    while text.endswith("\n"):
        text = text[:-1]
    return text


_title_re = re.compile(r"Chương\s*\d+[^<\n]*")


def extract_title(section_html: str) -> str:
    m = _title_re.search(section_html)
    if not m:
        return ""
    # Decode and tighten whitespace
    return html.unescape(_multi_ws_re.sub(" ", m.group(0)).strip())


def fetch_chapter(sess, chap_num: int):
    cid = chap_num + 1  # vntq off-by-one: chuongid=1 is preface, chuongid=2 = Chương 1
    raw = fetch_chuongid(sess, cid)
    parts = raw.split(SEPARATOR)
    if len(parts) < 4:
        raise RuntimeError(f"Unexpected response: {len(parts)} sections for chap {chap_num}")
    title = extract_title(parts[3]) or extract_title(parts[1])
    body_text = extract_chapter_text(parts[2])
    return {
        "chapter": chap_num,
        "chuongid": cid,
        "title": title,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "body_bytes": len(body_text.encode("utf-8")),
        "body": body_text,
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sess = make_session()
    print(f"Fetching chapters {args.start}..{args.end} from vietnamthuquan.eu (tuaid={NOVEL_ID})")
    print(f"Output dir: {args.out_dir}")

    successes = 0
    failures = []
    for chap in range(args.start, args.end + 1):
        body_path = args.out_dir / f"chap_{chap}.txt"
        meta_path = args.out_dir / f"chap_{chap}.meta.json"
        if body_path.exists() and meta_path.exists() and not args.overwrite:
            print(f"  [skip] chap {chap}: already crawled (use --overwrite to re-fetch)")
            continue
        try:
            ch = fetch_chapter(sess, chap)
        except Exception as e:
            print(f"  [FAIL] chap {chap}: {e}")
            failures.append(chap)
            continue

        body_path.write_text(ch["body"], encoding="utf-8")
        meta = {k: v for k, v in ch.items() if k != "body"}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [ok]   chap {chap:>2}  cid={ch['chuongid']:>3}  title={ch['title']!r}  {ch['body_bytes']}B")
        successes += 1
        if chap != args.end:
            time.sleep(args.sleep)

    print()
    print(f"Done: {successes} fetched, {len(failures)} failed")
    if failures:
        print(f"  failed chapters: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
