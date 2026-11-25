"""Per-chapter decisive win rates and chapter features for the thesis.

Run from the repository root (D:/Dynamic GraphRAG):
    python evaluation/per_chapter_analysis.py
Writes evaluation/analysis/per_chapter_rates.csv and per_chapter_features.csv
and prints both tables plus Spearman correlations.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
EVAL = ROOT / "evaluation"
JUDGE = EVAL / "llm_judge"
OUT = EVAL / "analysis"
GLOSSARY = ROOT / "data/knowledge_base/qwen3-8b-local_yes_hygiene_a2on/term_glossary.json"

CHAPS_A = [6, 7, 8, 9, 10]
CHAPS_B = list(range(20, 31))


def p(chaps, slug, judge):
    return [JUDGE / f"pairwise_chap{c}_{slug}_{judge}.csv" for c in chaps]


SPECS = [
    ("TM_on_vs_TM_off", "A", [JUDGE / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"]
     + p([7, 8, 9, 10], "TMon_vs_TMoff", "deepseek-chat")),
    ("Hygiene_vs_TM_on", "A", p(CHAPS_A, "Hygiene_vs_TMon", "deepseek-chat")),
    ("A2_vs_Hygiene", "A", p(CHAPS_A, "A2_vs_Hygiene", "deepseek-chat")),
    ("TM_on_vs_GG", "A", p(CHAPS_A, "TMon_vs_GG", "deepseek-chat")),
    ("Hygiene_vs_GG", "A", p(CHAPS_A, "Hygiene_vs_GG", "deepseek-chat")),
    ("A2_vs_GG", "A", p(CHAPS_A, "A2_vs_GG", "deepseek-chat")),
    ("RAG_BIG_vs_GG", "A", p(CHAPS_A, "Track3_RAGBIG_vs_GG", "deepseek-chat")),
    ("Qwen_ZS_vs_GG", "B", p(CHAPS_B, "QwenZS_vs_GG", "gemini35flash")),
    ("Qwen_random_fewshot_vs_GG", "B", p(CHAPS_B, "QwenRndFS_vs_GG", "gemini35flash")),
    ("Qwen_RAG_vs_GG", "B", p(CHAPS_B, "A2new_vs_GG", "gemini35flash")),
    ("DeepSeek_ZS_vs_GG", "B", p(CHAPS_B, "DeepSeekZS_vs_GG", "gemini35flash")),
    ("DeepSeek_RAG_vs_GG", "B", p(CHAPS_B, "DeepSeek_vs_GG", "gemini35flash")),
]


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def per_chapter_counts(paths):
    cnt = defaultdict(lambda: [0, 0, 0])
    for path in paths:
        if not path.exists():
            print("MISSING", path.name)
            continue
        for r in rows(path):
            ch = int(str(r.get("chapter", "")).strip())
            c = r.get("final_choice")
            if c == "a":
                cnt[ch][0] += 1
            elif c == "b":
                cnt[ch][1] += 1
            elif c == "tie":
                cnt[ch][2] += 1
    return cnt


def source_sentences(ch):
    d = "alignment_results" if ch <= 10 else "alignment_results_a2_20_30"
    path = EVAL / d / f"aligned_master_chap_{ch}.csv"
    return [r["Source_Text"] for r in rows(path) if (r.get("Source_Text") or "").strip()]


QUOTE = re.compile(r"[\"\u201c\u201d]")


def features(ch, terms):
    sents = source_sentences(ch)
    text = " ".join(sents)
    words = len(re.findall(r"[A-Za-z']+", text))
    dialogue = sum(1 for s in sents if QUOTE.search(s))
    low = text.lower()
    hits = 0
    for t in terms:
        hits += len(re.findall(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", low))
    return {
        "chapter": ch,
        "n_sent": len(sents),
        "n_words": words,
        "dialogue_pct": round(100 * dialogue / len(sents), 1),
        "term_per_100w": round(100 * hits / words, 2),
    }


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gloss = json.load(open(GLOSSARY, encoding="utf-8"))
    terms = sorted({k.strip().lower() for k in gloss if len(k.strip()) >= 3})
    print("glossary terms:", len(terms))

    feats = {ch: features(ch, terms) for ch in CHAPS_A + CHAPS_B}
    with open(OUT / "per_chapter_features.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(feats.values())).keys()))
        w.writeheader()
        for ch in sorted(feats):
            w.writerow(feats[ch])
            print(feats[ch])

    out_rows = []
    for name, tset, paths in SPECS:
        cnt = per_chapter_counts(paths)
        chs = sorted(cnt)
        rates = []
        for ch in chs:
            a, b, t = cnt[ch]
            rate = 100 * a / (a + b) if a + b else float("nan")
            rates.append(rate)
            out_rows.append({"comparison": name, "test_set": tset, "chapter": ch,
                             "a": a, "b": b, "tie": t, "n_dec": a + b,
                             "rate_pct": round(rate, 1)})
        dia = [feats[c]["dialogue_pct"] for c in chs]
        trm = [feats[c]["term_per_100w"] for c in chs]
        print(f"{name:28s} " + " ".join(f"{c}:{r:.1f}({cnt[c][0]+cnt[c][1]})" for c, r in zip(chs, rates)))
        print(f"{'':28s} min={min(rates):.1f} max={max(rates):.1f} "
              f"rho_dialogue={spearman(dia, rates):+.2f} rho_terms={spearman(trm, rates):+.2f}")

    with open(OUT / "per_chapter_rates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
