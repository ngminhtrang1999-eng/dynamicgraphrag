"""Flask side-by-side editor for chapter VN translations.

Workflow:
  1. `crawl_vntq.py` saves raw VN auto-translation to data/LOTM/chapters_vi_raw/chap_<N>.txt
  2. This app shows EN source (read-only) next to editable VN raw, lets you fix
     the MT errors paragraph-by-paragraph, then saves cleaned text to
     data/LOTM/chapters_vi/chap_<N>.txt (the "good" reference).
  3. Edited file then feeds into TM seed expansion and LoRA hold-out.

Run:
    python vntq_edit_app.py
Open http://localhost:5001/ in your browser.

Paths:
    data/LOTM/chapters/chap_<N>.txt          # EN source (if you have it)
    data/LOTM/chapters_vi_raw/chap_<N>.txt   # crawled, immutable
    data/LOTM/chapters_vi/chap_<N>.txt       # your edits — what the project uses
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for

ROOT = Path(__file__).parent
EN_DIR = ROOT / "data" / "LOTM" / "chapters"
# LOTM_RAW_DIR env var overrides the raw-VN source dir (e.g. chapters_vi_pdf_raw
# for PDF-derived text; vietnamthuquan crawling is banned as of 2026-06-10)
VI_RAW_DIR = ROOT / "data" / "LOTM" / os.environ.get("LOTM_RAW_DIR", "chapters_vi_pdf_raw")
VI_EDIT_DIR = ROOT / "data" / "LOTM" / "chapters_vi"

app = Flask(__name__)


def en_path(n):
    """EN source lives as chapter_<N>.txt (full-novel dump) or legacy chap_<N>.txt."""
    for name in (f"chapter_{n}.txt", f"chapter_{n:04d}.txt", f"chap_{n}.txt"):
        p = EN_DIR / name
        if p.exists():
            return p
    return EN_DIR / f"chap_{n}.txt"


def list_chapters():
    """Return [{n, en_exists, raw_exists, edited_exists, raw_bytes, edited_bytes}]."""
    chapters = set()
    if VI_RAW_DIR.exists():
        for p in VI_RAW_DIR.glob("chap_*.txt"):
            stem = p.stem.replace("chap_", "")
            if stem.isdigit():
                chapters.add(int(stem))
    if VI_EDIT_DIR.exists():
        for p in VI_EDIT_DIR.glob("chap_*.txt"):
            stem = p.stem.replace("chap_", "")
            if stem.isdigit():
                chapters.add(int(stem))
    out = []
    for n in sorted(chapters):
        en = en_path(n)
        raw = VI_RAW_DIR / f"chap_{n}.txt"
        edited = VI_EDIT_DIR / f"chap_{n}.txt"
        meta = VI_RAW_DIR / f"chap_{n}.meta.json"
        title = ""
        if meta.exists():
            try:
                title = json.loads(meta.read_text(encoding="utf-8")).get("title", "")
            except Exception:
                pass
        out.append({
            "n": n,
            "title": title,
            "en_exists": en.exists(),
            "raw_exists": raw.exists(),
            "edited_exists": edited.exists(),
            "raw_bytes": raw.stat().st_size if raw.exists() else 0,
            "edited_bytes": edited.stat().st_size if edited.exists() else 0,
        })
    return out


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_edit(n: int, text: str):
    VI_EDIT_DIR.mkdir(parents=True, exist_ok=True)
    (VI_EDIT_DIR / f"chap_{n}.txt").write_text(text, encoding="utf-8")


INDEX_TEMPLATE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>VNTQ Editor — Index</title>
<style>
body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; line-height: 1.55; }
h1 { font-size: 1.3rem; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }
th { background: #f3f3f3; font-weight: 600; font-size: 0.9rem; }
.status-edited { color: #28a745; font-weight: 600; }
.status-raw { color: #ff8c00; }
.status-no { color: #aaa; }
a.btn { display: inline-block; padding: 4px 12px; background: #4a90e2; color: white; text-decoration: none; border-radius: 4px; font-size: 0.85rem; }
</style>
</head><body>
<h1>Chapter VN editor</h1>
<p>EN source (if any): <code>data/LOTM/chapters/</code> · Raw VN MT: <code>data/LOTM/chapters_vi_raw/</code> · Your edits: <code>data/LOTM/chapters_vi/</code></p>
<table>
<thead><tr><th>Chap</th><th>Title</th><th>EN</th><th>Raw VN</th><th>Edited VN</th><th>Status</th><th></th></tr></thead>
<tbody>
{% for c in chapters %}
<tr>
<td>{{ c.n }}</td>
<td>{{ c.title or '-' }}</td>
<td>{% if c.en_exists %}<span class="status-edited">yes</span>{% else %}<span class="status-no">no</span>{% endif %}</td>
<td>{% if c.raw_exists %}{{ c.raw_bytes }}B{% else %}<span class="status-no">missing</span>{% endif %}</td>
<td>{% if c.edited_exists %}<span class="status-edited">{{ c.edited_bytes }}B</span>{% else %}<span class="status-raw">(none)</span>{% endif %}</td>
<td>
  {% if c.edited_exists %}<span class="status-edited">edited</span>
  {% elif c.raw_exists %}<span class="status-raw">raw only</span>
  {% else %}<span class="status-no">empty</span>{% endif %}
</td>
<td><a class="btn" href="{{ url_for('edit_chapter', n=c.n) }}">Open</a></td>
</tr>
{% endfor %}
</tbody></table>
{% if not chapters %}<p style="color:#888">No chapters crawled yet. Run <code>python crawl_vntq.py --start 11 --end 25</code> first.</p>{% endif %}
</body></html>
"""

EDIT_TEMPLATE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Edit chap {{ n }}</title>
<style>
body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 0; }
.topbar { background: #2c3e50; color: white; padding: 10px 18px; display: flex; align-items: center; gap: 16px; }
.topbar h1 { font-size: 1.05rem; margin: 0; flex: 1; }
.topbar a, .topbar button { color: white; text-decoration: none; padding: 6px 14px; border-radius: 4px; border: 1px solid white; background: transparent; cursor: pointer; font-size: 0.9rem; }
.topbar a:hover, .topbar button:hover { background: rgba(255,255,255,0.15); }
.topbar .saved { color: #95e095; font-size: 0.85rem; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 0; height: calc(100vh - 50px); }
.pane { padding: 12px 18px; overflow: auto; box-sizing: border-box; height: 100%; }
.pane.left { background: #f7f9fc; border-right: 1px solid #ddd; }
.pane.right { background: white; }
.pane h2 { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 8px; font-weight: 600; }
.pane.left pre { white-space: pre-wrap; font-family: -apple-system, "Segoe UI", sans-serif; font-size: 0.95rem; line-height: 1.6; margin: 0; }
.pane.right textarea { width: 100%; height: calc(100% - 50px); box-sizing: border-box; font-family: -apple-system, "Segoe UI", sans-serif; font-size: 0.95rem; line-height: 1.6; padding: 10px; border: 1px solid #ccc; border-radius: 4px; resize: none; }
.meta { font-size: 0.82rem; color: #888; margin-top: 6px; }
.no-source { color: #aaa; font-style: italic; padding: 20px; }
</style>
</head><body>
<form method="post" action="{{ url_for('save_chapter', n=n) }}" style="margin:0;">
<div class="topbar">
<h1>Chap {{ n }}{% if title %} — {{ title }}{% endif %}</h1>
{% if prev_n %}<a href="{{ url_for('edit_chapter', n=prev_n) }}">← Prev (chap {{ prev_n }})</a>{% endif %}
<a href="{{ url_for('index') }}">All chapters</a>
{% if next_n %}<a href="{{ url_for('edit_chapter', n=next_n) }}">Next (chap {{ next_n }}) →</a>{% endif %}
<button type="submit">💾 Save edits</button>
{% if saved %}<span class="saved">✓ Saved at {{ saved }}</span>{% endif %}
</div>
<div class="split">
<div class="pane left">
<h2>EN source — chap_{{ n }}.txt</h2>
{% if en_text %}
<pre>{{ en_text }}</pre>
{% else %}
<div class="no-source">
Chưa có file EN source cho chap {{ n }}.<br><br>
Drop EN text vào <code>data/LOTM/chapters/chap_{{ n }}.txt</code> rồi reload, hoặc edit VN-only.
</div>
{% endif %}
</div>
<div class="pane right">
<h2>VN — editable {% if edited_exists %}(loaded from your edits){% else %}(loaded from raw MT — first edit){% endif %}</h2>
<textarea name="vn_text" spellcheck="false">{{ vn_text }}</textarea>
<div class="meta">{{ vn_chars }} chars · {{ vn_lines }} lines · Ctrl+S in textarea = Save</div>
</div>
</div>
</form>
<script>
// Ctrl+S in textarea -> submit
document.querySelector("textarea").addEventListener("keydown", function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === "s") {
    e.preventDefault();
    document.querySelector("form").submit();
  }
});
</script>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_TEMPLATE, chapters=list_chapters())


@app.route("/chap/<int:n>")
def edit_chapter(n):
    en_text = load_text(en_path(n))
    edit_path = VI_EDIT_DIR / f"chap_{n}.txt"
    edited_exists = edit_path.exists()
    if edited_exists:
        vn_text = load_text(edit_path)
    else:
        vn_text = load_text(VI_RAW_DIR / f"chap_{n}.txt")

    title = ""
    meta = VI_RAW_DIR / f"chap_{n}.meta.json"
    if meta.exists():
        try:
            title = json.loads(meta.read_text(encoding="utf-8")).get("title", "")
        except Exception:
            pass

    chapters = list_chapters()
    chap_nums = [c["n"] for c in chapters]
    prev_n = next((m for m in reversed(chap_nums) if m < n), None)
    next_n = next((m for m in chap_nums if m > n), None)

    return render_template_string(
        EDIT_TEMPLATE,
        n=n, title=title,
        en_text=en_text, vn_text=vn_text,
        vn_chars=len(vn_text), vn_lines=vn_text.count("\n") + (1 if vn_text else 0),
        edited_exists=edited_exists, prev_n=prev_n, next_n=next_n,
        saved=request.args.get("saved", ""),
    )


@app.route("/chap/<int:n>/save", methods=["POST"])
def save_chapter(n):
    text = request.form.get("vn_text", "")
    save_edit(n, text)
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    return redirect(url_for("edit_chapter", n=n, saved=ts))


if __name__ == "__main__":
    VI_RAW_DIR.mkdir(parents=True, exist_ok=True)
    VI_EDIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"EN source dir:  {EN_DIR}")
    print(f"Raw VN dir:     {VI_RAW_DIR}")
    print(f"Edited VN dir:  {VI_EDIT_DIR}")
    # host=0.0.0.0 so the Windows browser can reach the Flask app running inside
    # WSL via http://localhost:5001 (WSL2 forwards localhost from Windows but
    # only when the service binds to all interfaces, not 127.0.0.1 inside WSL).
    print("Open http://localhost:5001/ in your Windows browser.")
    app.run(host="0.0.0.0", port=5001, debug=False)
