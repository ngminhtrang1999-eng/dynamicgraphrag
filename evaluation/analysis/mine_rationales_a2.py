"""A2 ablation: keyword frequency over judge rationales.

Adapted from mine_rationales.py. Two comparisons:
  - A2_vs_Hygiene (a=A2_on, b=Hygiene baseline) — marginal lift of coref
  - A2_vs_GG      (a=A2_on, b=Google Translate)  — residual gap after A2

Output: theme-delta tables (a_share - b_share); negative delta = theme dominant
on b-side. Used to attribute A2's lift (predicted: register_xung_ho; alternate
hypothesis: name_romanization or accuracy_mistrans via cleaner glossary filter).

Writes to evaluation/analysis/_a2_rationale_themes.log.
"""
from __future__ import annotations
import csv
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = ROOT / "evaluation" / "llm_judge"
LOG_PATH = Path(__file__).resolve().parent / "_a2_rationale_themes.log"

PAIRS_A2_VS_HYG = [JUDGE_DIR / f"pairwise_chap{c}_A2_vs_Hygiene_deepseek-chat.csv" for c in range(6, 11)]
PAIRS_A2_VS_GG  = [JUDGE_DIR / f"pairwise_chap{c}_A2_vs_GG_deepseek-chat.csv"     for c in range(6, 11)]

# Same inventory as mine_rationales.py — keeps results comparable cross-experiment.
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
    return {theme: re.compile("|".join(terms), flags=re.IGNORECASE)
            for theme, terms in KEYWORDS.items()}


def scan(paths, patterns):
    counts = collections.defaultdict(lambda: collections.Counter())
    chap_stats = collections.defaultdict(lambda: collections.Counter())
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


def report(label: str, paths, patterns, log):
    counts, chap_stats, n_rows = scan(paths, patterns)
    side_totals = collections.Counter()
    for c in chap_stats.values():
        side_totals.update(c)
    a_n, b_n, tie_n = side_totals["a_wins"], side_totals["b_wins"], side_totals["tie"]
    log.write(f"\n{'='*100}\n{label}   n={n_rows}   a={a_n} b={b_n} tie={tie_n}\n{'='*100}\n")
    log.write(f"{'theme':<28} {'a_wins':>8} {'b_wins':>8} {'tie':>6}   "
              f"{'a%':>6} {'b%':>6} {'tie%':>6}   {'delta':>8}\n")
    log.write("-" * 100 + "\n")
    rows = []
    for theme in KEYWORDS:
        c = counts[theme]
        a_share = c["a_wins"] / a_n if a_n else 0
        b_share = c["b_wins"] / b_n if b_n else 0
        tie_share = c["tie"] / tie_n if tie_n else 0
        rows.append((theme, c["a_wins"], c["b_wins"], c["tie"],
                     a_share, b_share, tie_share, a_share - b_share))
    rows.sort(key=lambda r: r[7])  # most negative delta first
    for theme, a, b, t, a_s, b_s, t_s, d in rows:
        log.write(f"{theme:<28} {a:>8} {b:>8} {t:>6}   "
                  f"{a_s*100:>5.1f}% {b_s*100:>5.1f}% {t_s*100:>5.1f}%   {d*100:+7.1f}pp\n")


def main():
    patterns = compile_patterns()
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write("A2 RATIONALE THEME MINING — 2026-05-16\n")
        log.write("Negative delta (a%-b%) = theme dominant on b-side (i.e. losing-side complaint)\n")
        report("A2_vs_Hygiene (a=A2_on, b=hygiene_on) — marginal A2 lift",
               PAIRS_A2_VS_HYG, patterns, log)
        report("A2_vs_GG (a=A2_on, b=GG) — residual gap after A2",
               PAIRS_A2_VS_GG, patterns, log)
    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
