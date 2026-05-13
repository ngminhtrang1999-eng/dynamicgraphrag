"""Alignment editor: EN chapter_NNNN.txt ↔ docln raw chapter ↔ curated VN.

The docln translation uses light-novel volume cuts that do not match the web
novel's chapter numbering 1:1. To produce a per-EN-chapter VN reference, the
user reads a docln chapter alongside an EN chapter, manually copies the slice
of VN that corresponds to the EN content, and saves it.

3-pane layout:
  LEFT   — EN source from data/LOTM/chapters/chapter_<NNNN>.txt   (read-only)
  MIDDLE — docln raw VN from data/LOTM/chapters_vi_docln_raw/chap_<M>.txt
           (read-only, selectable — user copies the matching slice)
  RIGHT  — editable textarea; on Save writes to
           data/LOTM/chapters_vi/chap_<en_chap>.txt (last-write-wins)

The MIDDLE pane has a dropdown of all available docln chapters so the user
can browse different volume cuts while keeping the same EN target on the LEFT.

Run:
    python docln_align_app.py
Open http://localhost:5002/ in your Windows browser.
"""

import datetime
import json
import re
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for

ROOT = Path(__file__).parent
EN_DIR = ROOT / "data" / "LOTM" / "chapters"
DOCLN_RAW_DIR = ROOT / "data" / "LOTM" / "chapters_vi_docln_raw"
VI_EDIT_DIR = ROOT / "data" / "LOTM" / "chapters_vi"

app = Flask(__name__)

_chap_num_re = re.compile(r"chapter_(\d+)\.txt$", re.IGNORECASE)


def list_en_chapters():
    if not EN_DIR.exists():
        return []
    nums = []
    for p in EN_DIR.glob("chapter_*.txt"):
        m = _chap_num_re.search(p.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def list_docln_chapters():
    out = []
    if not DOCLN_RAW_DIR.exists():
        return out
    for p in sorted(DOCLN_RAW_DIR.glob("chap_*.txt")):
        m = re.match(r"chap_(\d+)\.txt$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        title = ""
        meta_path = DOCLN_RAW_DIR / f"chap_{n}.meta.json"
        if meta_path.exists():
            try:
                title = json.loads(meta_path.read_text(encoding="utf-8")).get("title", "")
            except Exception:
                pass
        out.append({"n": n, "title": title, "bytes": p.stat().st_size})
    return out


def read_en(chap_num: int) -> str:
    p = EN_DIR / f"chapter_{chap_num:04d}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_docln(chap_num: int) -> str:
    p = DOCLN_RAW_DIR / f"chap_{chap_num}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_edited(en_chap: int) -> str:
    p = VI_EDIT_DIR / f"chap_{en_chap}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def save_edited(en_chap: int, text: str):
    VI_EDIT_DIR.mkdir(parents=True, exist_ok=True)
    (VI_EDIT_DIR / f"chap_{en_chap}.txt").write_text(text, encoding="utf-8")


INDEX_TEMPLATE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>docln alignment editor</title>
<style>
body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; line-height: 1.55; }
h1 { font-size: 1.3rem; }
.box { background: #f7f9fc; border: 1px solid #ddd; padding: 14px 18px; border-radius: 6px; margin: 14px 0; }
.box h2 { font-size: 0.95rem; margin: 0 0 8px; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 0.9rem; }
th { background: #f0f0f0; }
.btn { padding: 6px 16px; background: #4a90e2; color: white; text-decoration: none; border-radius: 4px; font-size: 0.9rem; border: none; cursor: pointer; }
.btn:hover { background: #357abd; }
input[type=number] { width: 80px; padding: 4px 8px; font-size: 0.95rem; }
.status-done { color: #28a745; font-weight: 600; }
.status-todo { color: #aaa; }
</style></head><body>
<h1>docln alignment editor</h1>
<p>Map a docln (light-novel) Vietnamese chapter onto an EN web-novel chapter. The numbering differs — you manually cut the matching slice.</p>

<div class="box">
<h2>Open an EN chapter to align</h2>
<form method="get" action="/align">
EN chapter number: <input type="number" name="en_chap" min="1" max="9999" value="11" required>
<button type="submit" class="btn">Open</button>
</form>
<p style="color:#666; font-size:0.85rem;">EN files available: {{ en_chapters_count }} ({{ en_first }} - {{ en_last }})</p>
</div>

<div class="box">
<h2>Available docln chapters (raw VN)</h2>
{% if docln_chapters %}
<table>
<tr><th>Chap</th><th>Title</th><th>Bytes</th></tr>
{% for d in docln_chapters %}
<tr><td>{{ d.n }}</td><td>{{ d.title or '-' }}</td><td>{{ d.bytes }}</td></tr>
{% endfor %}
</table>
<p style="color:#666; font-size:0.85rem;">Missing chapters (404 on web.archive): 8, 10, 12-17, 20.</p>
{% else %}
<p style="color:#999">No docln chapters crawled yet. Run <code>python crawl_docln.py</code>.</p>
{% endif %}
</div>

<div class="box">
<h2>Already-saved VN references</h2>
{% if edited %}
<table><tr><th>EN chap</th><th>Bytes</th></tr>
{% for e in edited %}<tr><td>{{ e.n }}</td><td>{{ e.bytes }}</td></tr>{% endfor %}
</table>
{% else %}
<p style="color:#999">No saved references yet. Output dir: <code>data/LOTM/chapters_vi/</code></p>
{% endif %}
</div>
</body></html>
"""

ALIGN_TEMPLATE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Align EN chap {{ en_chap }}</title>
<style>
body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 0; }
.topbar { background: #2c3e50; color: white; padding: 10px 18px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.topbar h1 { font-size: 1.0rem; margin: 0; flex: 1; min-width: 200px; }
.topbar a, .topbar button { color: white; text-decoration: none; padding: 5px 12px; border-radius: 4px; border: 1px solid white; background: transparent; cursor: pointer; font-size: 0.85rem; }
.topbar a:hover, .topbar button:hover { background: rgba(255,255,255,0.15); }
.topbar select, .topbar input[type=number] { color: #1a1a1a; padding: 4px 8px; border-radius: 4px; border: 1px solid #aaa; font-size: 0.85rem; }
.topbar .saved { color: #95e095; font-size: 0.85rem; }
.tri { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; height: calc(100vh - 50px); }
.pane { padding: 12px 16px; overflow: auto; box-sizing: border-box; height: 100%; border-right: 1px solid #ddd; }
.pane:last-child { border-right: 0; }
.pane h2 { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 8px; font-weight: 600; }
.pane.en { background: #fef9e6; }
.pane.docln { background: #f0f7ff; }
.pane.edit { background: #fff; }
.pane pre { white-space: pre-wrap; font-family: -apple-system, "Segoe UI", sans-serif; font-size: 0.92rem; line-height: 1.6; margin: 0; user-select: text; }
.pane textarea { width: 100%; height: calc(100% - 60px); box-sizing: border-box; font-family: -apple-system, "Segoe UI", sans-serif; font-size: 0.92rem; line-height: 1.6; padding: 8px; border: 1px solid #ccc; border-radius: 4px; resize: none; }
.meta { font-size: 0.78rem; color: #888; margin-top: 4px; }
.empty { color: #aaa; font-style: italic; padding: 20px; }
</style></head><body>

<div class="topbar">
<h1>EN chap {{ en_chap }}</h1>
EN: <input type="number" id="en-jump" value="{{ en_chap }}" min="1" max="9999">
<button onclick="location.href='/align?en_chap='+document.getElementById('en-jump').value">Go</button>
<a href="/align?en_chap={{ en_chap - 1 }}">← {{ en_chap - 1 }}</a>
<a href="/align?en_chap={{ en_chap + 1 }}">{{ en_chap + 1 }} →</a>
docln source:
<select id="docln-sel" onchange="document.getElementById('docln-frame').src='/docln/'+this.value">
{% for d in docln_chapters %}
<option value="{{ d.n }}" {% if d.n == current_docln %}selected{% endif %}>chap {{ d.n }}{% if d.title %} — {{ d.title }}{% endif %}</option>
{% endfor %}
</select>
<a href="/">All chapters</a>
{% if saved %}<span class="saved">✓ Saved at {{ saved }}</span>{% endif %}
</div>

<div class="tri">
<div class="pane en">
<h2>EN source — chapter_{{ '%04d' % en_chap }}.txt</h2>
{% if en_text %}<pre>{{ en_text }}</pre>{% else %}<div class="empty">No EN file for chap {{ en_chap }}.</div>{% endif %}
</div>

<div class="pane docln">
<h2>docln raw VN — chap {{ current_docln }} (read-only, select+copy)</h2>
{% if docln_text %}<pre>{{ docln_text }}</pre>{% else %}<div class="empty">No docln chap {{ current_docln }}. Pick another from the dropdown.</div>{% endif %}
</div>

<div class="pane edit">
<form method="post" action="/save/{{ en_chap }}" style="margin:0; height:100%;">
<h2>VN reference for EN chap {{ en_chap }} (paste cut content here)</h2>
<textarea name="vn_text" spellcheck="false" placeholder="Paste the matching VN slice from the middle pane, then edit if needed.">{{ edited_text }}</textarea>
<div style="margin-top:6px; display:flex; align-items:center; gap:10px;">
<button type="submit" class="btn" style="background:#4a90e2; color:white; border:none; padding:6px 14px; border-radius:4px; cursor:pointer; font-size:0.9rem;">💾 Save (Ctrl+S)</button>
<span class="meta">{{ edited_text|length }} chars · written to data/LOTM/chapters_vi/chap_{{ en_chap }}.txt</span>
</div>
</form>
</div>
</div>

<script>
// Hidden iframe trick is overkill — just navigate when select changes
document.getElementById('docln-sel').addEventListener('change', function() {
  const url = new URL(window.location.href);
  url.searchParams.set('docln', this.value);
  window.location.href = url.toString();
});

document.querySelector('textarea').addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    document.querySelector('form').submit();
  }
});
</script>
</body></html>
"""


@app.route("/")
def index():
    en_chaps = list_en_chapters()
    docln = list_docln_chapters()
    edited = []
    if VI_EDIT_DIR.exists():
        for p in sorted(VI_EDIT_DIR.glob("chap_*.txt")):
            m = re.match(r"chap_(\d+)\.txt$", p.name)
            if m:
                edited.append({"n": int(m.group(1)), "bytes": p.stat().st_size})
    return render_template_string(
        INDEX_TEMPLATE,
        en_chapters_count=len(en_chaps),
        en_first=en_chaps[0] if en_chaps else 0,
        en_last=en_chaps[-1] if en_chaps else 0,
        docln_chapters=docln,
        edited=edited,
    )


@app.route("/align")
def align():
    try:
        en_chap = int(request.args.get("en_chap", "11"))
    except ValueError:
        en_chap = 11
    docln_chapters = list_docln_chapters()
    # Default docln selection — first available, or user override
    docln_nums = [d["n"] for d in docln_chapters]
    try:
        current_docln = int(request.args.get("docln", str(docln_nums[0]) if docln_nums else "0"))
    except ValueError:
        current_docln = docln_nums[0] if docln_nums else 0

    en_text = read_en(en_chap)
    docln_text = read_docln(current_docln) if current_docln in docln_nums else ""
    edited_text = read_edited(en_chap)

    return render_template_string(
        ALIGN_TEMPLATE,
        en_chap=en_chap, current_docln=current_docln,
        en_text=en_text, docln_text=docln_text,
        edited_text=edited_text,
        docln_chapters=docln_chapters,
        saved=request.args.get("saved", ""),
    )


@app.route("/save/<int:en_chap>", methods=["POST"])
def save(en_chap):
    text = request.form.get("vn_text", "")
    save_edited(en_chap, text)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    docln = request.args.get("docln", "")
    qs = f"&docln={docln}" if docln else ""
    return redirect(f"/align?en_chap={en_chap}&saved={ts}{qs}")


if __name__ == "__main__":
    VI_EDIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"EN dir:        {EN_DIR}")
    print(f"docln raw dir: {DOCLN_RAW_DIR}")
    print(f"VN target dir: {VI_EDIT_DIR}")
    # 0.0.0.0 so the Windows browser reaches the WSL Flask app.
    print("Open http://localhost:5002/ in your Windows browser.")
    app.run(host="0.0.0.0", port=5002, debug=False)
