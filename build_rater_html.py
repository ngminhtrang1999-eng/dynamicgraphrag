"""Generate a single self-contained HTML rating app for non-technical raters.

Reads the BLIND sheet evaluation/human_eval/human_eval_pairs.csv, embeds only the
fields a rater may see (item_id, source, reference, candidate_A, candidate_B) --
never the key (which side is which system / the judge verdict) -- and writes a
zero-dependency offline page:

    evaluation/human_eval/rater_app.html

The rater double-clicks the file, types their name, judges 120 items with
mouse or keyboard (1=A, 2=tie, 3=B), and downloads
human_eval_rater_<name>.csv (columns item_id,choice) -- the exact format
score_human_eval.py consumes. Progress autosaves to the browser so they can
close and resume. No server, no install, no internet required.

Re-run after rebuilding the kit (build_human_eval_kit.py) to refresh the data.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
HE = ROOT / "evaluation" / "human_eval"
SRC = HE / "human_eval_pairs.csv"
OUT = HE / "rater_app.html"


def load_items():
    with open(SRC, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        items.append({
            "id": r["item_id"],
            "src": (r.get("source") or "").strip(),
            "ref": (r.get("reference") or "").strip(),
            "a": (r.get("candidate_A") or "").strip(),
            "b": (r.get("candidate_B") or "").strip(),
        })
    return items


HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Đánh giá bản dịch văn học</title>
<style>
  :root{
    --bg:#f4f1ea; --card:#fffdf8; --ink:#26221c; --muted:#8a8275;
    --line:#e3ddcf; --accent:#7a5c3e; --accent2:#3e5c7a; --tie:#6b6b6b;
    --ok:#2e7d4f;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:#26221c;
    font-family:"Iowan Old Style","Palatino Linotype",Georgia,"Times New Roman",serif;
    line-height:1.6;}
  .wrap{max-width:820px;margin:0 auto;padding:24px 18px 96px;}
  h1{font-size:1.5rem;margin:0 0 .2em}
  .sub{color:#6b6459;font-size:.95rem;margin-bottom:1.4em}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
    padding:22px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
  button{font-family:inherit;cursor:pointer}
  .btn{background:var(--accent);color:#fff;border:none;border-radius:8px;
    padding:11px 20px;font-size:1rem}
  .btn:hover{filter:brightness(1.07)}
  .btn.ghost{background:transparent;color:var(--accent);border:1px solid var(--accent)}
  input[type=text]{font-family:inherit;font-size:1.05rem;padding:10px 12px;
    border:1px solid var(--line);border-radius:8px;width:100%;background:#fff}
  /* progress */
  .topbar{position:sticky;top:0;background:var(--bg);padding:10px 0 12px;z-index:5;
    border-bottom:1px solid var(--line);margin-bottom:18px}
  .bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden}
  .bar > i{display:block;height:100%;background:var(--ok);width:0;transition:width .25s}
  .count{display:flex;justify-content:space-between;font-size:.9rem;color:#6b6459;
    margin-bottom:6px}
  .toolbar{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
  .mini{background:#efeadf;border:1px solid var(--line);border-radius:7px;
    padding:6px 11px;font-size:.85rem;color:#5c554a}
  .mini:hover{background:#e7e0d0}
  .savedhint{font-size:.82rem;color:#2e7d4f;margin-left:2px}
  /* segments */
  .seg{margin:0 0 14px}
  .seg .lab{font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;
    color:#9a917f;margin-bottom:4px}
  .seg .txt{font-size:1.12rem}
  .src{background:#f0ece1;border-left:3px solid #c9bfa6;padding:10px 14px;border-radius:6px}
  .ref{background:#eef1ee;border-left:3px solid #b7c9b7;padding:10px 14px;border-radius:6px}
  .cands{display:grid;grid-template-columns:1fr;gap:14px;margin-top:18px}
  .cand{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:#fff}
  .cand .tag{display:inline-block;font-size:.8rem;font-weight:700;color:#fff;
    background:#9a917f;border-radius:6px;padding:2px 9px;margin-bottom:8px}
  .choices{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:22px}
  .choice{padding:16px 10px;border:2px solid var(--line);border-radius:10px;
    background:#fff;font-size:1.02rem;text-align:center;transition:.12s}
  .choice:hover{border-color:var(--accent)}
  .choice.sel{border-color:var(--ok);background:#eaf5ee;font-weight:700}
  .choice .k{display:block;font-size:.74rem;color:#9a917f;margin-top:4px}
  .nav{display:flex;justify-content:space-between;align-items:center;margin-top:22px}
  .hide{display:none!important}
  .done-big{font-size:3rem;text-align:center;margin:.2em 0}
  .tips{color:#5c554a;font-size:.95rem;padding-left:1.2em;margin:.3em 0}
  .tips li{margin:.35em 0}
  .rubricbox{background:#f6f2e8;border:1px solid var(--line);border-radius:10px;
    padding:14px 16px 6px;margin:10px 0 4px}
  .rubricbox ol{margin:.2em 0;padding-left:1.3em}
  .remind{font-size:.82rem;color:#8a8275;background:#f6f2e8;border:1px solid var(--line);
    border-radius:8px;padding:7px 11px;margin-top:18px;line-height:1.45}
  .warn{color:#a4532b;font-size:.92rem;margin-top:10px}
  .namebox{display:flex;gap:10px;margin:18px 0;align-items:center}
  .footer-note{font-size:.85rem;color:#8a8275;margin-top:14px;text-align:center}
</style>
</head>
<body>
<div class="wrap">

  <!-- ============ INTRO ============ -->
  <div id="intro" class="panel">
    <h1>Đánh giá chất lượng bản dịch văn học</h1>
    <div class="sub">Anh/chị sẽ so sánh hai bản dịch tiếng Việt của cùng một câu và chọn bản nào hay hơn. Khoảng <b>__COUNT__ câu</b>, làm trong ~2–3 giờ (có thể tạm dừng, mở lại làm tiếp).</div>
    <div class="rubricbox">
      <b>Tiêu chí chấm — xét theo thứ tự ưu tiên từ trên xuống:</b>
      <ol class="tips">
        <li><b>Đủ ngữ nghĩa.</b> Bản dịch có thể ngắn hơn hoặc gộp câu, miễn truyền đạt <b>đủ ý</b> câu gốc. Bản nào <b>thiếu hoặc sai ý</b> → thua ngay.</li>
        <li><b>Xưng hô đúng &amp; nhất quán.</b> Cách nhân vật gọi nhau (anh/em/ngài/tiểu thư…) phải hợp vai vế và <b>giữ nhất quán</b> xuyên suốt.</li>
        <li><b>Không lẫn chữ lạ.</b> Không sót chữ Trung/Anh giữa câu; tên riêng phải <b>phiên âm Hán-Việt đúng</b> (vd “Chu Minh Thụy”, không để “Zhou Mingrui”).</li>
      </ol>
      Hai bản <b>ngang nhau</b> ở cả 3 tiêu chí → chọn bản <b>đọc mượt/tự nhiên hơn</b>; vẫn ngang → bấm <b>Ngang nhau</b>.
    </div>
    <ul class="tips">
      <li>Mỗi màn hiện: <b>câu gốc tiếng Anh</b>, <b>bản tham khảo</b> (nếu có), và <b>Bản A / Bản B</b>.</li>
      <li>Không cần biết bản nào của hệ thống nào — chọn theo cảm nhận người đọc.</li>
      <li>Phím tắt: <b>1</b>=Bản A, <b>2</b>=Ngang nhau, <b>3</b>=Bản B, <b>←</b>=quay lại.</li>
    </ul>
    <div class="namebox">
      <input type="text" id="name" placeholder="Tên/biệt danh của anh chị (vd: minh, rater2)" autocomplete="off">
    </div>
    <button class="btn" id="start">Bắt đầu →</button>
    <button class="btn ghost" id="resume" style="margin-left:8px">📂 Mở tiến độ đã lưu</button>
    <div class="warn hide" id="nameWarn">Vui lòng nhập tên trước khi bắt đầu.</div>
    <div class="footer-note" style="text-align:left;margin-top:16px">⚠️ Quan trọng: để khỏi mất bài, thỉnh thoảng bấm <b>💾 Lưu tiến độ</b> để tải file <code>.json</code> về máy. Lần sau mở lại, bấm <b>📂 Mở tiến độ</b> rồi chọn file đó là làm tiếp được.</div>
  </div>

  <!-- ============ RATING ============ -->
  <div id="rate" class="hide">
    <div class="topbar">
      <div class="count"><span id="pos"></span><span id="answered"></span></div>
      <div class="bar"><i id="fill"></i></div>
      <div class="toolbar">
        <button class="mini" id="saveProg">💾 Lưu tiến độ (.json)</button>
        <button class="mini" id="openProg">📂 Mở tiến độ</button>
        <button class="mini" id="expCsv">⬇ Xuất CSV ngay</button>
        <input type="file" id="impFile" accept=".json,application/json" class="hide">
        <span class="savedhint" id="savedHint"></span>
      </div>
    </div>
    <div class="panel">
      <div class="seg"><div class="lab">Câu gốc (English)</div><div class="txt src" id="src"></div></div>
      <div class="seg" id="refSeg"><div class="lab">Bản tham khảo</div><div class="txt ref" id="ref"></div></div>
      <div class="cands">
        <div class="cand"><span class="tag">Bản A</span><div class="txt" id="ca"></div></div>
        <div class="cand"><span class="tag">Bản B</span><div class="txt" id="cb"></div></div>
      </div>
      <div class="remind">Ưu tiên: <b>1)</b> đủ ngữ nghĩa (thiếu ý → thua) → <b>2)</b> xưng hô đúng &amp; nhất quán → <b>3)</b> không lẫn chữ Trung/Anh, tên phiên âm đúng. Ngang cả 3 → chọn bản mượt hơn.</div>
      <div class="choices">
        <button class="choice" data-v="A">🅰 Bản A hay hơn<span class="k">phím 1</span></button>
        <button class="choice" data-v="tie">⚖ Ngang nhau<span class="k">phím 2</span></button>
        <button class="choice" data-v="B">🅱 Bản B hay hơn<span class="k">phím 3</span></button>
      </div>
      <div class="nav">
        <button class="btn ghost" id="prev">← Quay lại</button>
        <button class="btn ghost" id="next">Bỏ qua →</button>
      </div>
    </div>
    <div class="footer-note">Tiến độ tự lưu trên máy anh/chị. Có thể đóng và mở lại file này để làm tiếp.</div>
  </div>

  <!-- ============ DONE ============ -->
  <div id="done" class="hide panel">
    <div class="done-big">✅</div>
    <h1 style="text-align:center">Hoàn thành!</h1>
    <div class="sub" id="doneMsg" style="text-align:center"></div>
    <div style="text-align:center;margin:18px 0">
      <button class="btn" id="dl">⬇ Tải file kết quả (CSV)</button>
      <button class="btn ghost" id="dlj" style="margin-left:8px">Tải dạng JSON</button>
    </div>
    <div class="footer-note">Gửi lại file <b>human_eval_rater_…csv</b> vừa tải về cho người phụ trách.<br>Muốn sửa? Bấm nút bên dưới để xem lại.</div>
    <div style="text-align:center;margin-top:14px">
      <button class="btn ghost" id="review">↩ Xem lại / sửa câu trả lời</button>
    </div>
  </div>

</div>

<script>
const DATA = __DATA__;
let rater = "", idx = 0, answers = {};

const $ = s => document.querySelector(s);
function storeKey(){ return "humaneval_" + rater.toLowerCase(); }
function save(){ try{ localStorage.setItem(storeKey(), JSON.stringify(answers)); }catch(e){} }
function load(){ try{ return JSON.parse(localStorage.getItem(storeKey())||"{}"); }catch(e){ return {}; } }
function answeredCount(){ return DATA.filter(d => answers[d.id]).length; }

function show(view){
  $("#intro").classList.add("hide");
  $("#rate").classList.add("hide");
  $("#done").classList.add("hide");
  $(view).classList.remove("hide");
}

function render(){
  const d = DATA[idx];
  $("#pos").textContent = "Câu " + (idx+1) + " / " + DATA.length;
  $("#answered").textContent = "Đã chọn: " + answeredCount() + "/" + DATA.length;
  $("#fill").style.width = (answeredCount()/DATA.length*100) + "%";
  $("#src").textContent = d.src;
  if (d.ref){ $("#refSeg").classList.remove("hide"); $("#ref").textContent = d.ref; }
  else { $("#refSeg").classList.add("hide"); }
  $("#ca").textContent = d.a;
  $("#cb").textContent = d.b;
  document.querySelectorAll(".choice").forEach(b=>{
    b.classList.toggle("sel", answers[d.id] === b.dataset.v);
  });
  $("#prev").disabled = (idx === 0);
  window.scrollTo(0,0);
}

function pick(v){
  const d = DATA[idx];
  answers[d.id] = v;
  save();
  if (idx < DATA.length - 1){ idx++; render(); }
  else { maybeFinish(); }
}

function maybeFinish(){
  if (answeredCount() === DATA.length){ finish(); }
  else {
    // jump to first unanswered
    const i = DATA.findIndex(d => !answers[d.id]);
    if (i >= 0){ idx = i; render(); }
  }
}

function finish(){
  $("#doneMsg").textContent = "Anh/chị đã đánh giá " + answeredCount() + "/" + DATA.length + " câu.";
  show("#done");
}

function csvCell(s){ s = String(s); return /[",\r\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; }
function toCSV(){
  const rows = [["item_id","choice"]];
  DATA.forEach(d => rows.push([d.id, answers[d.id]||""]));
  return rows.map(r => r.map(csvCell).join(",")).join("\r\n");
}
function safeName(){ return (rater||"rater").replace(/[^a-zA-Z0-9_-]/g,"_").toLowerCase(); }
function download(text, fname, mime){
  const blob = new Blob([text], {type: mime+";charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = fname; document.body.appendChild(a); a.click();
  setTimeout(()=>{ document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
}

function flash(msg){
  const h = $("#savedHint"); if (!h) return;
  h.textContent = msg;
  setTimeout(() => { if (h.textContent === msg) h.textContent = ""; }, 4000);
}
function exportProgress(){
  download(JSON.stringify({rater: rater, answers: answers}, null, 2),
           "tiendo_" + safeName() + ".json", "application/json");
  flash("Đã lưu tiến độ. Giữ file .json này để mở lại sau.");
}
function importProgressFile(file){
  const r = new FileReader();
  r.onload = () => {
    try{
      const obj = JSON.parse(r.result);
      const inc = obj.answers || obj;           // accept {rater,answers} or raw map
      Object.assign(answers, inc);
      if (obj.rater && !rater) rater = obj.rater;
      if (!rater) rater = "rater";
      save();
      idx = DATA.findIndex(d => !answers[d.id]); if (idx < 0) idx = 0;
      show("#rate"); render();
      flash("Đã nhập tiến độ: " + answeredCount() + "/" + DATA.length + " câu.");
    }catch(err){ alert("File tiến độ không đọc được (cần đúng file .json đã lưu)."); }
  };
  r.readAsText(file);
}

// ---- intro ----
$("#start").onclick = () => {
  const n = $("#name").value.trim();
  if (!n){ $("#nameWarn").classList.remove("hide"); return; }
  rater = n;
  answers = load();
  idx = DATA.findIndex(d => !answers[d.id]); if (idx < 0) idx = 0;
  show("#rate"); render();
};
$("#name").addEventListener("keydown", e => { if (e.key === "Enter") $("#start").click(); });
$("#resume").onclick = () => $("#impFile").click();

// ---- progress save/open/export (available during rating) ----
$("#saveProg").onclick = exportProgress;
$("#expCsv").onclick    = () => download(toCSV(), "human_eval_rater_" + safeName() + ".csv", "text/csv");
$("#openProg").onclick  = () => $("#impFile").click();
$("#impFile").onchange  = (e) => { const f = e.target.files[0]; if (f) importProgressFile(f); e.target.value = ""; };

// ---- rating ----
document.querySelectorAll(".choice").forEach(b => b.onclick = () => pick(b.dataset.v));
$("#prev").onclick = () => { if (idx>0){ idx--; render(); } };
$("#next").onclick = () => { if (idx < DATA.length-1){ idx++; render(); } else maybeFinish(); };
document.addEventListener("keydown", e => {
  if ($("#rate").classList.contains("hide")) return;
  if (e.key === "1") pick("A");
  else if (e.key === "2") pick("tie");
  else if (e.key === "3") pick("B");
  else if (e.key === "ArrowLeft") $("#prev").click();
  else if (e.key === "ArrowRight") $("#next").click();
});

// ---- done ----
$("#dl").onclick  = () => download(toCSV(), "human_eval_rater_"+safeName()+".csv", "text/csv");
$("#dlj").onclick = () => download(JSON.stringify({rater:rater, answers:answers}, null, 2),
                                  "human_eval_rater_"+safeName()+".json", "application/json");
$("#review").onclick = () => { idx = 0; show("#rate"); render(); };
</script>
</body>
</html>
"""


def main():
    items = load_items()
    data_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__DATA__", data_json).replace("__COUNT__", str(len(items)))
    OUT.write_text(html, encoding="utf-8")
    nref = sum(1 for it in items if it["ref"])
    print(f"wrote {OUT}  ({len(items)} items, {nref} with reference)")


if __name__ == "__main__":
    main()
