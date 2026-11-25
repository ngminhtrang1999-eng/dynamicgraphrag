"""Track 1 step 2-3: keyword frequency over judge rationales.

Loads all 9 pairwise CSVs (TMon vs GG, TMon vs TMoff for chap 6-10) and counts
how often each Vietnamese rubric-aligned keyword appears in rationale_1 +
rationale_2, broken down by final_choice (a-wins / b-wins / tie).

For TMon_vs_GG: a = RAG+TM, b = GG.
For TMon_vs_TMoff: a = RAG+TM, b = RAG only (no TM).

The keyword inventory targets axes the judge prompt enumerates plus failure
modes we've observed in smoke tests (code-switch, Hán-Việt leak, entity tag
leak, hallucination, missing translation).
"""
from __future__ import annotations
import csv
import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = ROOT / "evaluation" / "llm_judge"

PAIRS_GG = [JUDGE_DIR / f"pairwise_chap{c}_TMon_vs_GG_deepseek-chat.csv" for c in range(6, 11)]
PAIRS_TM = [JUDGE_DIR / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"] + [
    JUDGE_DIR / f"pairwise_chap{c}_TMon_vs_TMoff_deepseek-chat.csv" for c in range(7, 11)
]

# Themes -> regex alternation. Word boundaries are tricky for Vietnamese
# (diacritics + spaces inside compound words), so we use lowercase substring
# matching with a tolerant set of surface forms per theme.
KEYWORDS = {
    "fluency_natural": [r"tự nhiên", r"mượt", r"trôi chảy", r"trau chuốt", r"văn phong tốt", r"đọc hay"],
    "fluency_awkward": [r"không tự nhiên", r"khô khan", r"cứng", r"vụng về", r"lủng củng", r"khó hiểu", r"khó đọc"],
    "register_xung_ho": [r"xưng hô", r"đại từ", r"anh ta", r"hắn", r"y\b", r"gã", r"ông ta", r"bà ta", r"cô ta"],
    "accuracy_mistrans": [r"dịch sai", r"sai nghĩa", r"hiểu nhầm", r"không sát", r"lệch nghĩa", r"hiểu sai", r"sai ngữ pháp"],
    "accuracy_omission": [r"bỏ sót", r"thiếu", r"không dịch", r"thiếu thông tin", r"bỏ qua"],
    "accuracy_addition": [r"thêm thắt", r"thừa", r"hư cấu", r"bịa", r"thêm chi tiết", r"thêm từ"],
    "terminology_hanviet": [r"hán việt", r"hán-việt", r"chữ hán", r"từ hán"],
    "terminology_consistency": [r"thuật ngữ", r"nhất quán", r"không nhất quán", r"thống nhất", r"khác bản trước"],
    "code_switch": [r"code-switch", r"code switching", r"tiếng anh", r"giữ nguyên tiếng anh", r"english", r"chen tiếng"],
    "entity_leak": [r"\[entity", r"\[person", r"\[org", r"thẻ", r"tag", r"chú thích"],
    "name_romanization": [r"phiên âm", r"romaniz", r"tên riêng", r"chu minh", r"zhou ming", r"klein", r"không phiên"],
    "style_literary": [r"văn học", r"văn chương", r"chất văn", r"giàu hình ảnh", r"thi vị", r"hấp dẫn"],
    "style_flat": [r"khô", r"không có cảm xúc", r"thiếu cảm xúc", r"thiếu sinh động", r"đơn điệu"],
    "cultural": [r"văn hóa", r"phù hợp với người việt", r"thuần việt", r"không phù hợp"],
    "punctuation": [r"dấu câu", r"chấm phẩy", r"ngắt câu", r"câu dài"],
    "structure": [r"cấu trúc", r"song song", r"trật tự", r"đảo ngữ"],
    "grammar": [r"ngữ pháp", r"sai cấu trúc", r"câu què", r"câu cụt"],
    "redflag_critical": [r"red flag", r"nghiêm trọng", r"lỗi nặng", r"vulgar", r"thô tục", r"lồn", r"loạn luân"],
}


def compile_patterns():
    out = {}
    for theme, terms in KEYWORDS.items():
        joined = "|".join(terms)
        out[theme] = re.compile(joined, flags=re.IGNORECASE)
    return out


def scan(paths, label, patterns):
    """Returns: per-theme counter of which-side, per-chapter stats, sample lookup."""
    counts = collections.defaultdict(lambda: collections.Counter())  # theme -> {a_wins, b_wins, tie}
    chap_stats = collections.defaultdict(lambda: collections.Counter())  # chap -> {a, b, tie}
    rows_seen = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows_seen += 1
                fc = row["final_choice"].strip().lower()
                key = {"a": "a_wins", "b": "b_wins", "tie": "tie"}.get(fc, "other")
                chap_stats[row["chapter"]][key] += 1
                text = (row.get("rationale_1") or "") + " \n " + (row.get("rationale_2") or "")
                text_lc = text.lower()
                for theme, pat in patterns.items():
                    if pat.search(text_lc):
                        counts[theme][key] += 1
    return counts, chap_stats, rows_seen


def fmt_pct(num, den):
    return f"{num/den*100:.1f}%" if den else "n/a"


def main():
    patterns = compile_patterns()
    for label, paths in [("TMon_vs_GG (a=RAG+TM, b=GG)", PAIRS_GG),
                         ("TMon_vs_TMoff (a=RAG+TM, b=RAG_no_TM)", PAIRS_TM)]:
        counts, chap_stats, n_rows = scan(paths, label, patterns)
        # Aggregate side totals
        a_total = sum(c["a_wins"] for c in counts.values())  # not meaningful; need totals from chap_stats
        side_totals = collections.Counter()
        for chap, c in chap_stats.items():
            side_totals.update(c)
        a_n = side_totals["a_wins"]
        b_n = side_totals["b_wins"]
        tie_n = side_totals["tie"]
        print(f"\n{'='*78}\n{label}   n={n_rows}   a={a_n} b={b_n} tie={tie_n}\n{'='*78}")
        print(f"{'theme':<28} {'a_wins':>10} {'b_wins':>10} {'tie':>8}   {'a%':>6} {'b%':>6} {'tie%':>6}   delta")
        print("-" * 100)
        # Sort by absolute (a_share - b_share) to surface most-discriminating themes
        rows = []
        for theme in KEYWORDS:
            c = counts[theme]
            a_share = c["a_wins"] / a_n if a_n else 0
            b_share = c["b_wins"] / b_n if b_n else 0
            tie_share = c["tie"] / tie_n if tie_n else 0
            delta = a_share - b_share
            rows.append((theme, c["a_wins"], c["b_wins"], c["tie"], a_share, b_share, tie_share, delta))
        rows.sort(key=lambda r: r[7])  # negative delta = b-side dominated theme (e.g. GG)
        for theme, a, b, t, a_s, b_s, t_s, d in rows:
            print(f"{theme:<28} {a:>10} {b:>10} {t:>8}   {a_s*100:>5.1f}% {b_s*100:>5.1f}% {t_s*100:>5.1f}%   {d*100:+.1f}pp")


if __name__ == "__main__":
    main()
