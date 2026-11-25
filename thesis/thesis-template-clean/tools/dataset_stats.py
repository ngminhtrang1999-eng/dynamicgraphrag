#!/usr/bin/env python
"""Thống kê bộ dữ liệu cho mục 3.2 / Phụ lục D của luận văn (THESIS_REVISION_PLAN.md §4.2).

Script CHỈ ĐỌC dữ liệu trong repo "Dynamic GraphRAG"; không ghi gì vào repo.
Kết quả được ghi ra tools/dataset_stats_output.md (cạnh script này).

Cách chạy (môi trường conda `graphrag` trong WSL, nơi đã chạy thực nghiệm):
    python thesis/thesis-template-clean/tools/dataset_stats.py [--labse]

    --repo PATH   thư mục gốc repo (mặc định: ba cấp trên thư mục tools/)
    --labse       tính thêm (cần sentence-transformers + LaBSE, nên có GPU):
                  (a) phân bố cosine LaBSE giữa nhóm câu nguồn và bản đối chiếu đã căn chỉnh;
                  (b) độ tương đồng top-1 khi truy xuất TM cho từng đoạn 500 token của tập A, B
                      (tái hiện TranslationMemory.retrieve với k=3, τ=0,30).

Các quy ước đếm:
  * Câu: `smart_sentence_split` sao chép nguyên văn từ align_ablation.py (bộ tách dùng khi căn chỉnh).
    Mẫu tách chỉ nhận chữ hoa ASCII sau dấu câu, nên với tiếng Việt cũng báo thêm số câu theo
    chữ hoa Unicode (cột "câu*").
  * Từ (EN) / âm tiết (VI): số token tách theo khoảng trắng sau khi gộp khoảng trắng.
  * Token: tiktoken cl100k_base — đúng bộ đếm mà utils.split_text_with_overlap dùng để chia đoạn.
  * Đoạn: semchunk.chunk(text, 500 | 700, token_counter=cl100k) như core/pipeline.py, sau khi
    gộp khoảng trắng như main.py.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys
import unicodedata
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED = "<MERGED_ABOVE>"
TM_CH = [1] + list(range(11, 20))
SET_A = list(range(6, 11))
SET_B = list(range(20, 31))
KB_MAIN = "qwen3-8b-local_yes_hygiene_a2on"


def role(n):
    if n in SET_A:
        return "Tập A"
    if n in SET_B:
        return "Tập B"
    if n in TM_CH:
        return "TM"
    return "không dùng"


# ---------------------------------------------------------------- helpers
def smart_sentence_split(text):  # verbatim from align_ablation.py
    text = str(text).replace("…", "...").replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    pattern = r'(?<=[.?!])\s+(?=[A-Z"“0-9])'
    return [s.strip() for s in re.split(pattern, text) if len(s.strip()) > 1]


def unicode_sentence_split(text):
    text = re.sub(r"\s+", " ", str(text).replace("…", "...")).strip()
    parts = re.split(r'(?<=[.?!])\s+(?=["“‘\'(\[]?[^\W\d_a-zà-ỹđ]|\d)', text)
    return [s for s in parts if len(s.strip()) > 1]


def read(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        return f.read()


def collapse(t):
    return re.sub(r"\s+", " ", t).strip()


def nwords(t):
    return len(collapse(t).split())


def toks(s):
    return re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())


def shingles(t, n=6):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def fmt(x, d=0):
    if isinstance(x, str):
        return x
    if x is None:
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",")  # 1 234,5 (kiểu Việt)


class Out:
    def __init__(self):
        self.lines = []

    def h(self, level, text):
        self.lines += ["", "#" * level + " " + text, ""]

    def p(self, text=""):
        self.lines.append(text)

    def table(self, header, rows, latex=False, caption=None):
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        if latex:
            esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace(" ", r"\,")
            self.lines += ["", "```latex", r"\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}", r"\toprule",
                           " & ".join(r"\textbf{" + esc(c) + "}" for c in header) + r" \\", r"\midrule"]
            for r in rows:
                self.lines.append(" & ".join(esc(c) for c in r) + r" \\")
            self.lines += [r"\bottomrule", r"\end{tabular}", "```"]
        self.lines.append("")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    ap.add_argument("--labse", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "dataset_stats_output.md"))
    args = ap.parse_args()
    R = args.repo
    L = os.path.join(R, "data", "LOTM")
    J = lambda *a: os.path.join(R, *a)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ntok = lambda t: len(enc.encode(t))
        tok_note = f"tiktoken {tiktoken.__version__} cl100k_base"
    except Exception:
        enc, ntok, tok_note = None, None, "tiktoken không có — bỏ cột token"
    try:
        import semchunk
        chunker = (lambda t, n: semchunk.chunk(t, chunk_size=n, token_counter=ntok)) if ntok else None
    except Exception:
        chunker = None

    o = Out()
    o.p("# Thống kê bộ dữ liệu (sinh tự động)")
    o.p("")
    o.p(f"Sinh bởi `tools/dataset_stats.py` lúc {datetime.now():%Y-%m-%d %H:%M}; repo: `{R}`; "
        f"bộ đếm token: {tok_note}; semchunk: {'có' if chunker else 'không'}; LaBSE: {'có' if args.labse else 'không (chạy với --labse)'}.")
    o.p("Không chỉnh tay file này — chạy lại script để cập nhật. Vai trò chương: TM = nguồn bộ nhớ dịch (1, 11–19); "
        "Tập A = 6–10; Tập B = 20–30; 2–5 không dùng trong thực nghiệm cuối.")

    # ---------- T1 + T2: EN source and VI reference per chapter
    ref_path = lambda n: os.path.join(L, "translations", "reference", f"ref_{n}.txt") if n <= 10 \
        else os.path.join(L, "chapters_vi", f"chap_{n}.txt")
    en_path = lambda n: os.path.join(L, "chapters", f"chapter_{n:04d}.txt")

    rows_en, rows_vi, tot = [], [], collections.Counter()
    per = {}
    for n in range(1, 31):
        en = read(en_path(n))
        vi = read(ref_path(n)) if os.path.exists(ref_path(n)) else ""
        e_s, e_w, e_c = len(smart_sentence_split(en)), nwords(en), len(collapse(en))
        e_t = ntok(collapse(en)) if ntok else None
        e_ch = len(chunker(collapse(en), 500)) if chunker else None
        e_cm = len(chunker(collapse(en), 700)) if chunker else None
        v_s, v_s2, v_w, v_c = len(smart_sentence_split(vi)), len(unicode_sentence_split(vi)), nwords(vi), len(collapse(vi))
        per[n] = dict(e_s=e_s, e_w=e_w, v_w=v_w)
        rows_en.append([n, role(n), e_s, fmt(e_w), fmt(e_t), fmt(e_c), e_ch if e_ch is not None else "–", e_cm if e_cm is not None else "–"])
        rows_vi.append([n, role(n), os.path.relpath(ref_path(n), L).replace("\\", "/"), v_s, v_s2, fmt(v_w), fmt(v_c),
                        fmt(v_w / e_w, 2) if e_w else "–", fmt(v_c / e_c, 2) if e_c else "–"])
        for k, v in dict(e_s=e_s, e_w=e_w, e_t=e_t or 0, e_c=e_c, e_ch=e_ch or 0, e_cm=e_cm or 0,
                         v_s=v_s, v_s2=v_s2, v_w=v_w, v_c=v_c).items():
            tot[(role(n), k)] += v
            tot[("all", k)] += v

    o.h(2, "T1. Văn bản nguồn tiếng Anh, chương 1–30")
    o.table(["Chương", "Vai trò", "Câu", "Từ", "Token", "Ký tự", "Đoạn 500", "Đoạn 700"], rows_en)
    sum_rows = []
    for g in ["TM", "Tập A", "Tập B", "không dùng", "all"]:
        sum_rows.append([g if g != "all" else "Tổng 1–30", fmt(tot[(g, 'e_s')]), fmt(tot[(g, 'e_w')]), fmt(tot[(g, 'e_t')]),
                         fmt(tot[(g, 'e_ch')]), fmt(tot[(g, 'v_s')]), fmt(tot[(g, 'v_w')]),
                         fmt(tot[(g, 'v_w')] / tot[(g, 'e_w')], 2) if tot[(g, 'e_w')] else "–"])
    o.h(3, "T1-tổng. Tổng hợp theo vai trò (bảng chính cho mục 3.2.7)")
    o.table(["Nhóm", "Câu EN", "Từ EN", "Token EN", "Đoạn 500", "Câu VI", "Âm tiết VI", "VI/EN"], sum_rows, latex=True)

    o.h(2, "T2. Bản đối chiếu tiếng Việt, chương 1–30")
    o.p("`câu` = bộ tách gốc (chữ hoa ASCII); `câu*` = tách theo chữ hoa Unicode. `VI/EN (từ)` = âm tiết VI / từ EN.")
    o.table(["Chương", "Vai trò", "Tệp", "Câu", "Câu*", "Âm tiết", "Ký tự", "VI/EN (từ)", "VI/EN (ký tự)"], rows_vi)

    # ---------- T2b: source attribution of VI references (6-gram containment)
    pdf_p = J("_pdf_repaired.txt")
    o.h(2, "T2b. Nguồn của bản đối chiếu (độ phủ 6-gram so với các nguồn thô)")
    if os.path.exists(pdf_p):
        pdf = shingles(toks(read(pdf_p)))
        docln = {int(re.search(r"(\d+)", os.path.basename(p)).group(1)): shingles(toks(read(p)))
                 for p in glob.glob(os.path.join(L, "chapters_vi_docln_raw", "chap_*.txt"))}
        docln_all = set().union(*docln.values()) if docln else set()
        vntq_all = set().union(*[shingles(toks(read(p))) for p in glob.glob(os.path.join(L, "chapters_vi_raw", "chap_*.txt"))])
        rows = []
        for n in range(1, 31):
            s = shingles(toks(read(ref_path(n))))
            c = lambda b: len(s & b) / max(1, len(s))
            best = max(docln.items(), key=lambda kv: len(s & kv[1]), default=(None, set()))
            rows.append([n, fmt(c(pdf), 2), fmt(c(docln_all), 2),
                         f"LN {best[0]}" if best[0] is not None and c(best[1]) > 0.03 else "–", fmt(c(vntq_all), 2)])
        o.p("Tỷ lệ 6-gram (từ, chữ thường, NFC) của bản đối chiếu xuất hiện trong: PDF \"LOTM – Việt Hóa\" đã sửa "
            "(`_pdf_repaired.txt`), toàn bộ bản crawl docln (Kouji), toàn bộ bản crawl vietnamthuquan. "
            "Giá trị ≈1 nghĩa là bản đối chiếu được lấy nguyên văn từ nguồn đó.")
        o.table(["Chương", "PDF", "docln", "docln (chương LN)", "vntq"], rows)
    else:
        o.p("Bỏ qua: không thấy `_pdf_repaired.txt`.")

    # ---------- T3: alignment distribution
    align_dirs = [(SET_A, "alignment_results_qwen3-8b-local-a2"), (TM_CH[1:], "alignment_results_reference"),
                  (SET_B, "alignment_results_a2_20_30")]
    o.h(2, "T3. Phân bố kiểu căn chỉnh (cột `Source_Text`/`ref` của aligned_master_chap_N.csv)")
    o.p("Mỗi dòng CSV là một câu nguồn. Nhóm = một dòng có `ref` thật cùng các dòng `<MERGED_ABOVE>` theo sau "
        "(k:1, k câu nguồn chung một câu đích). 1:k (nhiều câu VI gộp vào một câu nguồn) ước lượng bằng số câu "
        "Unicode trong ô `ref` > 1. `Không ref` = câu nguồn không có bản đối chiếu.")
    rows, groups_by_ch = [], {}
    gt = collections.Counter()
    for chs, d in align_dirs:
        for n in chs:
            p = J("evaluation", d, f"aligned_master_chap_{n}.csv")
            if not os.path.exists(p):
                rows.append([n, d, "thiếu", "", "", "", "", ""]); continue
            with open(p, encoding="utf-8-sig") as f:
                data = list(csv.DictReader(f))
            n_src = len(data)
            empty = sum(1 for r in data if not (r.get("ref") or "").strip())
            merged = sum(1 for r in data if (r.get("ref") or "").strip() == MERGED)
            groups, cur = [], None
            for r in data:
                s, t = (r.get("Source_Text") or "").strip(), (r.get("ref") or "").strip()
                if t and t != MERGED:
                    cur = [[s], t]; groups.append(cur)
                elif t == MERGED and cur is not None:
                    cur[0].append(s)
            groups_by_ch[n] = [(" ".join(g[0]), g[1]) for g in groups]
            k1 = sum(1 for g in groups if len(g[0]) > 1)
            one_k = sum(1 for g in groups if len(unicode_sentence_split(g[1])) > 1)
            one_one = sum(1 for g in groups if len(g[0]) == 1 and len(unicode_sentence_split(g[1])) <= 1)
            rows.append([n, d.replace("alignment_results_", ""), n_src, len(groups), one_one, k1, one_k, empty])
            for k, v in dict(src=n_src, grp=len(groups), o=one_one, k1=k1, ok=one_k, e=empty, m=merged).items():
                gt[(role(n), k)] += v
    o.table(["Chương", "Thư mục", "Câu nguồn", "Nhóm", "1:1", "k:1", "1:k (ước)", "Không ref"], rows)
    o.h(3, "T3-tổng")
    o.table(["Nhóm", "Câu nguồn", "Nhóm câu", "1:1", "k:1", "1:k (ước)", "Không ref", "% không ref"],
            [[g, gt[(g, 'src')], gt[(g, 'grp')], gt[(g, 'o')], gt[(g, 'k1')], gt[(g, 'ok')], gt[(g, 'e')],
              fmt(100 * gt[(g, 'e')] / max(1, gt[(g, 'src')]), 1)] for g in ["TM", "Tập A", "Tập B"]], latex=True)

    # ---------- T4: translation memory
    o.h(2, "T4. Bộ nhớ dịch (TM)")
    kb = J("data", "knowledge_base")
    tm_rows = []
    for p in sorted(glob.glob(os.path.join(kb, "*", "translation_memory.json"))):
        d = json.load(open(p, encoding="utf-8"))
        tm_rows.append([os.path.basename(os.path.dirname(p)), len(d),
                        datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")])
    o.p("Các bản TM trong `data/knowledge_base/*/` (cùng 928 mục; bản `_tm` là bản trước khi gói làm sạch sửa "
        "6 mục chương 1 \"Zhou Mingrui\" → \"Chu Minh Thụy\").")
    o.table(["Thư mục KB", "Số mục", "Sửa lần cuối"], tm_rows)
    tm = json.load(open(os.path.join(kb, KB_MAIN, "translation_memory.json"), encoding="utf-8"))
    # attribute entries to chapters via aligned CSV groups (+ chapter-1 CSV used for the original seed)
    key = lambda s, t: (collapse(s).lower(), collapse(t).lower())
    lookup = {}
    for n, gs in groups_by_ch.items():
        for s, t in gs:
            lookup.setdefault(key(s, t), set()).add(n)
    for p in glob.glob(J("evaluation", "alignment_results_*", "aligned_master_chap_1.csv")):
        with open(p, encoding="utf-8-sig") as f:
            cur = None
            for r in csv.DictReader(f):
                s, t = (r.get("Source_Text") or "").strip(), (r.get("ref") or "").strip()
                if t and t != MERGED:
                    if cur: lookup.setdefault(key(" ".join(cur[0]), cur[1]), set()).add(1)
                    cur = [[s], t]
                elif cur is not None and s:
                    cur[0].append(s)
            if cur: lookup.setdefault(key(" ".join(cur[0]), cur[1]), set()).add(1)
    en_text = {n: collapse(read(en_path(n))).lower() for n in range(1, 31)}
    by_ch, last = collections.Counter(), None
    src_len, tgt_len = collections.defaultdict(list), collections.defaultdict(list)
    for x in tm:
        hits = lookup.get(key(x["source"], x["translation"]), set())
        if len(hits) != 1:
            probe = collapse(x["source"]).lower()[:80]
            en_hits = {n for n, t in en_text.items() if probe in t}
            hits = hits & en_hits if hits & en_hits else (en_hits if len(en_hits) == 1 else hits)
        ch = min(hits) if len(hits) == 1 else (last if last in hits else (min(hits) if hits else None))
        last = ch
        by_ch[ch] += 1
        src_len[ch].append(nwords(x["source"])); tgt_len[ch].append(nwords(x["translation"]))
    rows = [[c if c is not None else "?", by_ch[c], fmt(statistics.mean(src_len[c]), 1), fmt(statistics.mean(tgt_len[c]), 1)]
            for c in sorted(by_ch, key=lambda c: (c is None, c or 0))]
    allsrc = [v for l in src_len.values() for v in l]; alltgt = [v for l in tgt_len.values() for v in l]
    rows.append(["Tổng", len(tm), fmt(statistics.mean(allsrc), 1), fmt(statistics.mean(alltgt), 1)])
    o.p(f"Phân bố mục TM (`{KB_MAIN}`) theo chương nguồn, ghép với nhóm câu trong CSV căn chỉnh và văn bản EN:")
    o.table(["Chương nguồn", "Số cặp", "Từ EN/mục (TB)", "Âm tiết VI/mục (TB)"], rows, latex=True)

    # top-1 similarity from run logs
    sims = collections.defaultdict(list)
    for p in glob.glob(J("logs", "*.log")) + glob.glob(J("*.log")):
        cur = None
        for line in open(p, encoding="utf-8", errors="ignore"):
            m = re.search(r"PROCESSING CHAPTER (\d+) \[QWEN3-8B-LOCAL", line)
            if m: cur = int(m.group(1)); continue
            if "PROCESSING CHAPTER" in line: cur = None; continue
            m = re.search(r"TM: (\d+) examples retrieved \(top sim=([\d.]+)\)", line)
            if m and cur is not None:
                sims[(os.path.basename(p), cur)].append((int(m.group(1)), float(m.group(2))))
    if sims:
        rows = []
        for (f, ch), v in sorted(sims.items()):
            s = [x[1] for x in v]
            rows.append([f, ch, len(v), fmt(statistics.mean(s), 3), fmt(min(s), 3), fmt(max(s), 3), fmt(statistics.mean(x[0] for x in v), 2)])
        o.h(3, "T4b. Độ tương đồng top-1 khi truy xuất TM, ghi trong log chạy Qwen3-8B")
        o.p("Chỉ có log cho tập A (`logs/a2on_run.log`, `logs/ablation_chap6_tm.log`); lượt cosine của tập B không lưu log "
            "(xem T4c khi chạy với --labse).")
        o.table(["Log", "Chương", "Đoạn", "Top-1 TB", "Min", "Max", "Số ví dụ TB"], rows)

    # ---------- T5: glossary by type
    o.h(2, "T5. Bảng thuật ngữ theo loại (trạng thái cuối của từng thư mục KB)")
    types = ["PERSON", "DEITY", "ORG", "LOCATION", "CONCEPT", "ITEM", "TITLE", "UNKNOWN"]
    rows = []
    for d in ["qwen3-8b-local_yes_tm", "qwen3-8b-local_yes_notm", "qwen3-8b-local_yes_hygiene", KB_MAIN,
              KB_MAIN + "_rndfs", "deepseek-chat_yes_hygiene"]:
        p = os.path.join(kb, d, "term_glossary.json")
        if not os.path.exists(p): continue
        g = json.load(open(p, encoding="utf-8"))
        c = collections.Counter((v.get("type") or "UNKNOWN") if isinstance(v, dict) else "UNKNOWN" for v in g.values())
        other = sum(v for k, v in c.items() if k not in types)
        rows.append([d] + [c.get(t, 0) for t in types] + [other, len(g)])
    o.table(["Thư mục KB"] + types + ["khác", "Tổng"], rows, latex=True)
    o.p("Ghi chú: mục từ không lưu chương xuất hiện, nên chưa tính được số mục tăng theo chương từ file này.")

    # ---------- T6: per-chapter features (already used in Appendix B)
    p = J("evaluation", "analysis", "per_chapter_features.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        o.h(2, "T6. Đặc trưng chương (sao lại từ evaluation/analysis/per_chapter_features.csv)")
        o.table(list(data[0].keys()), [list(r.values()) for r in data])

    # ---------- T7: PDF repair
    o.h(2, "T7. Sửa văn bản PDF \"LOTM – Việt Hóa\"")
    raw_p = J("_pymupdf_full.txt")
    if os.path.exists(raw_p) and os.path.exists(pdf_p):
        eth = lambda t: sum(1 for ch in t if "ሀ" <= ch <= "᎟")
        raw, rep = read(raw_p), read(pdf_p)
        er, ep = eth(raw), eth(rep)
        o.table(["Đại lượng", "Trước sửa (`_pymupdf_full.txt`)", "Sau sửa (`_pdf_repaired.txt`)"],
                [["Ký tự", fmt(len(raw)), fmt(len(rep))],
                 ["Ký tự khối Ethiopic (U+1200–U+139F)", fmt(er), fmt(ep)],
                 ["Token theo khoảng trắng", fmt(len(raw.split())), fmt(len(rep.split()))]])
        o.p(f"Số glyph Ethiopic được giải mã: {fmt(er - ep)} ({fmt(100 * (er - ep) / max(1, er), 1)}%). "
            f"Chênh số token theo khoảng trắng ({fmt(len(raw.split()) - len(rep.split()))}) là cận dưới của số âm tiết bị tách được nối lại.")
    else:
        o.p("Bỏ qua: thiếu `_pymupdf_full.txt` hoặc `_pdf_repaired.txt`.")

    # ---------- T8: crawl provenance
    o.h(2, "T8. Thu thập bản tiếng Việt (siêu dữ liệu crawl)")
    rows = []
    for d, label in [("chapters_vi_docln_raw", "docln.net qua web.archive.org"), ("chapters_vi_raw", "vietnamthuquan.eu"),
                     ("chapters_vi_pdf_raw", "PDF LOTM – Việt Hóa")]:
        metas = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(L, d, "*.meta.json"))]
        txts = glob.glob(os.path.join(L, d, "chap_*.txt"))
        when = sorted(str(m.get("fetched_at") or m.get("extracted_at") or m.get("created_at") or "") for m in metas)
        when = [w for w in when if w]
        chs = sorted(int(re.search(r"(\d+)", os.path.basename(p)).group(1)) for p in txts)
        rows.append([label, f"`{d}/`", len(txts), f"{chs[0]}–{chs[-1]}" if chs else "–",
                     f"{when[0][:10]} → {when[-1][:10]}" if when else "–"])
    o.table(["Nguồn", "Thư mục", "Số tệp", "Số hiệu chương", "Ngày thu thập"], rows)
    o.p("Số hiệu chương của docln là chương light novel (mỗi chương LN ≈ 2–3 chương web), xem cột \"docln (chương LN)\" ở T2b.")

    # ---------- optional LaBSE analyses
    if args.labse:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/LaBSE")
        emb = lambda xs: model.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        o.h(2, "T9. Cosine LaBSE giữa nhóm câu nguồn và bản đối chiếu đã căn chỉnh")
        bins = [0, .3, .4, .5, .6, .7, .8, .9, 1.01]
        rows = []
        for g in ["TM", "Tập A", "Tập B"]:
            pairs = [p for n, gs in groups_by_ch.items() if role(n) == g for p in gs]
            if not pairs: continue
            a, b = emb([p[0] for p in pairs]), emb([p[1] for p in pairs])
            cs = (a * b).sum(1)
            h, _ = np.histogram(cs, bins=bins)
            rows.append([g, len(cs), fmt(float(cs.mean()), 3), fmt(float(np.median(cs)), 3)] + [int(x) for x in h])
        o.table(["Nhóm", "Cặp", "TB", "Trung vị"] + [f"[{bins[i]:.1f},{min(bins[i+1],1):.1f})" for i in range(len(bins) - 1)], rows)

        o.h(2, "T4c. Tái hiện truy xuất TM (k=3, τ=0,30) cho từng đoạn 500 token")
        if chunker is None:
            o.p("Bỏ qua: thiếu tiktoken/semchunk.")
        else:
            E = np.load(os.path.join(kb, KB_MAIN, "translation_memory.npy")).astype(np.float32)
            rows = []
            for g, chs in [("Tập A", SET_A), ("Tập B", SET_B)]:
                top1, nk = [], []
                for n in chs:
                    chunks = chunker(collapse(read(en_path(n))), 500)
                    q = emb(chunks)
                    s = q @ E.T
                    srt = np.sort(s, axis=1)[:, ::-1]
                    top1 += list(srt[:, 0]); nk += list((srt[:, :3] >= 0.30).sum(1))
                top1, nk = np.array(top1), np.array(nk)
                rows.append([g, len(top1), fmt(float(top1.mean()), 3), fmt(float(np.median(top1)), 3),
                             fmt(float(top1.min()), 3), fmt(float(top1.max()), 3), fmt(float(nk.mean()), 2),
                             int((nk == 0).sum())])
            o.p("Mô phỏng `TranslationMemory.retrieve` trên đoạn chia bằng semchunk (500 token cl100k) của văn bản đã gộp "
                "khoảng trắng; dùng `translation_memory.npy` của " f"`{KB_MAIN}`.")
            o.table(["Tập", "Đoạn", "Top-1 TB", "Trung vị", "Min", "Max", "Số ví dụ ≥ τ (TB)", "Đoạn không có ví dụ"], rows, latex=True)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(o.lines).strip() + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
