"""Flask web UI for Phase 3 human pairwise review of MT candidates.

Run:
    python review_app.py
Then open http://localhost:5000 in your browser.

Reads:  evaluation/llm_judge/spot_check_v2_pairs.csv
Saves:  evaluation/llm_judge/spot_check_v2_responses.csv (per-pair choice + notes + ts)

The UI emphasizes the rubric-aligned framing: look for word-level errors and
xưng hô first, then prose flow. This is the explicit fix to Phase 2's prose-bias
spot-check that scored 14.3% agreement.
"""

import csv
import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "evaluation" / "llm_judge"
PAIRS_PATH = EVAL_DIR / "spot_check_v2_pairs.csv"
RESPONSES_PATH = EVAL_DIR / "spot_check_v2_responses.csv"

app = Flask(__name__)


def load_pairs():
    with open(PAIRS_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_responses():
    if not RESPONSES_PATH.exists():
        return {}
    out = {}
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["set_name"], r["chapter"], r["sentence_id"])] = r
    return out


def save_response(set_name, chap, sid, choice, notes):
    """Append a response. If the same pair was answered before, the LATEST row
    in the CSV wins on score_spot_check_v2 read (it reads last-write-wins)."""
    new_file = not RESPONSES_PATH.exists()
    with open(RESPONSES_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "set_name", "chapter", "sentence_id", "choice", "notes", "timestamp"
        ])
        if new_file:
            w.writeheader()
        w.writerow({
            "set_name": set_name, "chapter": chap, "sentence_id": sid,
            "choice": choice, "notes": notes,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })


TEMPLATE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Phase 3 Review — Dynamic GraphRAG</title>
<style>
body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 920px; margin: 24px auto; padding: 0 16px; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 1.3rem; margin: 0 0 8px; }
.bar { background: #eee; border-radius: 4px; height: 6px; margin: 8px 0 18px; overflow: hidden; }
.bar > div { background: #4a90e2; height: 100%; transition: width .2s; }
.progress { font-size: 0.85rem; color: #555; }
details { background: #fff8e1; border: 1px solid #f0d674; padding: 10px 14px; border-radius: 6px; margin: 12px 0; }
details summary { font-weight: 600; cursor: pointer; }
details ol { margin: 8px 0 0 18px; }
details li { margin: 4px 0; }
.pair { border: 1px solid #ddd; border-radius: 6px; padding: 14px 18px; margin: 14px 0; background: #fff; }
.pair h2 { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 4px; font-weight: 600; }
.pair .text { font-size: 1.0rem; margin-bottom: 6px; white-space: pre-wrap; }
.src { background: #f5f5f5; }
.ref { background: #e8f4f8; }
.cand { background: #fff; border: 1.5px solid #aaa; }
.cand-a { border-color: #4a90e2; }
.cand-b { border-color: #e2754a; }
.choices label { display: inline-block; margin-right: 24px; padding: 6px 16px; border: 1.5px solid #aaa; border-radius: 4px; cursor: pointer; }
.choices input[type=radio] { margin-right: 6px; }
textarea { width: 100%; box-sizing: border-box; min-height: 50px; font-family: inherit; font-size: 0.95rem; padding: 6px; }
.btn { padding: 8px 18px; font-size: 1.0rem; cursor: pointer; border-radius: 4px; border: 1px solid #4a90e2; background: #4a90e2; color: white; margin-right: 8px; }
.btn.secondary { background: #fff; color: #4a90e2; }
.nav { margin-top: 14px; }
.done-banner { background: #d4edda; border: 1px solid #28a745; color: #155724; padding: 18px; border-radius: 6px; text-align: center; }
.prior-note { background: #fff3cd; border: 1px solid #ffeaa7; padding: 8px 12px; border-radius: 4px; margin: 10px 0; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>Phase 3 Review — Dynamic GraphRAG (Pair {{ idx_1based }}/{{ total }})</h1>
<div class="bar"><div style="width: {{ progress_pct }}%"></div></div>
<div class="progress">Đã trả lời: {{ answered }}/{{ total }} · Pair hiện tại: {{ pair.set_name }} chap {{ pair.chapter }} sid {{ pair.sentence_id }}</div>

<details {% if idx_1based == 1 %}open{% endif %}>
<summary>HƯỚNG DẪN (đọc trước khi đánh giá — quan trọng cho Phase 3)</summary>
<p>Đánh giá 2 bản dịch theo <b>rubric</b> (không phải gut feel). Lần trước miss "Lồn" và "ông ta" — lần này focus vào word-level errors trước, prose flow sau.</p>
<ol>
<li><b>Lỗi word-level (RED FLAG — auto pick bản còn lại):</b>
    <ul>
    <li>Từ tục tĩu / sai chính tả tiếng Việt (vd "Lồn" thay vì "Loen", "absurđ" thay vì "vô lý")</li>
    <li>Code-switching tiếng Anh giữa câu tiếng Việt</li>
    <li>Bịa thông tin lớn / mất chi tiết then chốt</li>
    </ul>
</li>
<li><b>Xưng hô (lỗi nặng):</b>
    <ul><li>Đại từ sai địa vị/tuổi nhân vật (vd "ông" cho nhân vật trẻ). Nếu một bản đúng, một bản sai → chọn bản đúng.</li></ul>
</li>
<li><b>Văn phong:</b>
    <ul><li>Câu chảy mượt, đúng văn phong tiểu thuyết Việt. Không khô cứng "máy dịch".</li></ul>
</li>
<li><b>Tie khi nào:</b>
    <ul>
    <li>Cả hai cùng cấp về 3 tiêu chí trên.</li>
    <li>Khác biệt chỉ là nuance từ vựng nhỏ (vd "cười" vs "cười nhẹ") — không đáng kể.</li>
    </ul>
</li>
</ol>
<p><b>QUY TRÌNH ĐỌC:</b> (1) Quét nhanh tìm word-level errors ở cả A và B. (2) Check xưng hô. (3) Nếu hai bản đều sạch lỗi, mới so sánh văn phong. (4) Chọn A/B/Tie + ghi notes nếu có nhận xét.</p>
</details>

{% if prior %}
<div class="prior-note">Bạn đã trả lời pair này trước: <b>{{ prior.choice }}</b>{% if prior.notes %} (notes: {{ prior.notes }}){% endif %}. Submit lại sẽ ghi đè.</div>
{% endif %}

<form method="post" action="{{ url_for('submit', idx=idx) }}">

<div class="pair src">
<h2>Source (English)</h2>
<div class="text">{{ pair.source }}</div>
</div>

<div class="pair ref">
<h2>Reference (Vietnamese gold)</h2>
<div class="text">{{ pair.reference if pair.reference else '(no reference for this row — MERGED_ABOVE)' }}</div>
</div>

<div class="pair cand cand-a">
<h2>Bản dịch A</h2>
<div class="text">{{ pair.candidate_A }}</div>
</div>

<div class="pair cand cand-b">
<h2>Bản dịch B</h2>
<div class="text">{{ pair.candidate_B }}</div>
</div>

<div class="choices">
<label><input type="radio" name="choice" value="A" {% if prior and prior.choice == 'A' %}checked{% endif %}>A hay hơn</label>
<label><input type="radio" name="choice" value="B" {% if prior and prior.choice == 'B' %}checked{% endif %}>B hay hơn</label>
<label><input type="radio" name="choice" value="tie" {% if prior and prior.choice == 'tie' %}checked{% endif %}>Tie / tương đương</label>
</div>

<p><b>Notes (tùy chọn — ghi nhanh lý do nếu có):</b></p>
<textarea name="notes" placeholder="Vd: B có lỗi 'Lồn'; A xưng hô sai...">{{ prior.notes if prior else '' }}</textarea>

<div class="nav">
<button type="submit" class="btn">Submit &amp; Next</button>
{% if idx > 0 %}<a href="{{ url_for('show_pair', idx=idx-1) }}" class="btn secondary">← Prev</a>{% endif %}
{% if idx < total - 1 %}<a href="{{ url_for('show_pair', idx=idx+1) }}" class="btn secondary">Skip →</a>{% endif %}
</div>

</form>

</body>
</html>
"""

DONE_TEMPLATE = """<!doctype html>
<html lang="vi">
<head><meta charset="utf-8"><title>Done</title>
<style>body{font-family:sans-serif;max-width:720px;margin:60px auto;padding:0 16px;text-align:center;line-height:1.6;}
.done-banner{background:#d4edda;border:1px solid #28a745;color:#155724;padding:24px;border-radius:6px;}
a{display:inline-block;margin-top:16px;}</style></head>
<body>
<div class="done-banner">
<h1>Hoàn thành {{ answered }}/{{ total }} pairs!</h1>
<p>Responses đã lưu tại <code>{{ responses_path }}</code></p>
<p>Session sau em sẽ chạy <code>python score_spot_check_v2.py</code> để tính agreement mới vs deepseek-chat judge.</p>
</div>
<a href="{{ url_for('show_pair', idx=0) }}">← Quay lại pair đầu để review</a>
</body>
</html>
"""


@app.route("/")
def index():
    pairs = load_pairs()
    responses = load_responses()
    for i, p in enumerate(pairs):
        key = (p["set_name"], p["chapter"], p["sentence_id"])
        if key not in responses:
            return redirect(url_for("show_pair", idx=i))
    # All done
    return render_template_string(
        DONE_TEMPLATE,
        answered=len(responses),
        total=len(pairs),
        responses_path=str(RESPONSES_PATH.relative_to(ROOT)),
    )


@app.route("/pair/<int:idx>")
def show_pair(idx):
    pairs = load_pairs()
    if idx < 0 or idx >= len(pairs):
        return redirect(url_for("index"))
    pair = pairs[idx]
    responses = load_responses()
    key = (pair["set_name"], pair["chapter"], pair["sentence_id"])
    prior = responses.get(key)
    answered = len(responses)
    total = len(pairs)
    progress_pct = round(100 * answered / total) if total else 0
    return render_template_string(
        TEMPLATE,
        pair=pair, idx=idx, idx_1based=idx + 1, total=total,
        answered=answered, progress_pct=progress_pct, prior=prior,
    )


@app.route("/submit/<int:idx>", methods=["POST"])
def submit(idx):
    pairs = load_pairs()
    if idx < 0 or idx >= len(pairs):
        return redirect(url_for("index"))
    pair = pairs[idx]
    choice = request.form.get("choice", "").strip().lower()
    notes = request.form.get("notes", "").strip()
    if choice in {"a", "b", "tie"}:
        save_response(pair["set_name"], pair["chapter"], pair["sentence_id"], choice, notes)
    # Move to next pair (or done page)
    next_idx = idx + 1
    if next_idx >= len(pairs):
        return redirect(url_for("index"))
    return redirect(url_for("show_pair", idx=next_idx))


if __name__ == "__main__":
    print(f"Loading pairs from: {PAIRS_PATH}")
    print(f"Saving responses to: {RESPONSES_PATH}")
    print("Open http://localhost:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
