"""Aggregate paper statistics from existing judge and human-eval CSVs.

Outputs:
    evaluation/paper_stats_summary.csv
"""

from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
JUDGE = EVAL / "llm_judge"
HUMAN = EVAL / "human_eval"
OUT = EVAL / "paper_stats_summary.csv"

LABELS = ("A", "B", "tie")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return 100 * (center - half), 100 * (center + half)


def sign_p_observed_direction(a: int, b: int) -> float:
    """One-sided exact sign-test p-value in the observed direction."""
    n = a + b
    if n == 0:
        return float("nan")
    k = min(a, b)
    # CDF P(X <= k), X~Bin(n, .5). Log-sum-exp avoids underflow for n>1000.
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    return min(math.exp(m) * sum(math.exp(x - m) for x in logs), 1.0)


def fmt_float(x: float, digits: int = 6) -> str:
    if math.isnan(x):
        return ""
    return f"{x:.{digits}g}"


def judge_row(label: str, paths: list[Path], group: str) -> dict[str, str]:
    a = b = tie = n = 0
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(path.name)
            continue
        rows = read_csv(path)
        n += len(rows)
        a += sum(1 for r in rows if r.get("final_choice") == "a")
        b += sum(1 for r in rows if r.get("final_choice") == "b")
        tie += sum(1 for r in rows if r.get("final_choice") == "tie")
    dec = a + b
    rate = 100 * a / dec if dec else float("nan")
    lo, hi = wilson(a, dec)
    odds = a / b if b else float("inf") if a else float("nan")
    return {
        "kind": "judge",
        "group": group,
        "comparison": label,
        "n_total": str(n),
        "n_dec": str(dec),
        "a": str(a),
        "b": str(b),
        "tie": str(tie),
        "rate_pct": f"{rate:.1f}" if not math.isnan(rate) else "",
        "ci_low": f"{lo:.1f}" if not math.isnan(lo) else "",
        "ci_high": f"{hi:.1f}" if not math.isnan(hi) else "",
        "p_sign": fmt_float(sign_p_observed_direction(a, b)),
        "diff_from_50": f"{rate - 50:.1f}" if not math.isnan(rate) else "",
        "odds_ratio": fmt_float(odds),
        "notes": "; ".join(f"missing {m}" for m in missing),
    }


def paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(JUDGE.glob(pat)))
    return out


def chap_paths(chaps: range | list[int], slug: str, judge: str) -> list[Path]:
    return [JUDGE / f"pairwise_chap{c}_{slug}_{judge}.csv" for c in chaps]


def norm_choice(choice: str | None) -> str | None:
    c = (choice or "").strip().lower()
    if c == "a":
        return "A"
    if c == "b":
        return "B"
    if c == "tie":
        return "tie"
    return None


def cohen_kappa(xs: list[str | None], ys: list[str | None]) -> tuple[float, int, int]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x and y]
    n = len(pairs)
    if not n:
        return float("nan"), 0, 0
    agree = sum(1 for x, y in pairs if x == y)
    cnt_x = defaultdict(int)
    cnt_y = defaultdict(int)
    for x, y in pairs:
        cnt_x[x] += 1
        cnt_y[y] += 1
    po = agree / n
    pe = sum((cnt_x[l] / n) * (cnt_y[l] / n) for l in LABELS)
    return ((po - pe) / (1 - pe) if pe != 1 else float("nan")), agree, n


def human_choice_to_system(choice: str | None, keyrow: dict[str, str]) -> str | None:
    if choice is None:
        return None
    if choice == "tie":
        return "tie"
    side_is = keyrow["A_is"] if choice == "A" else keyrow["B_is"]
    return "a" if side_is == "CAND" else "b"


def load_human() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str | None]]]:
    key_path = HUMAN / "human_eval_key.csv"
    if not key_path.exists():
        return {}, {}
    key = {r["item_id"]: r for r in read_csv(key_path)}
    raters = {}
    for path in sorted(HUMAN.glob("human_eval_rater_*.csv")):
        rows = {r["item_id"]: norm_choice(r.get("choice")) for r in read_csv(path)}
        if any(rows.values()):
            # public alias by sorted filename (matches score_human_eval.py: Rater 1, Rater 2, ...)
            raters[f"rater{len(raters) + 1}"] = rows
    return key, raters


def human_rows() -> list[dict[str, str]]:
    key, raters = load_human()
    if not key or not raters:
        return []
    rows: list[dict[str, str]] = []
    item_ids = sorted(key, key=lambda x: int(x))

    for left, right in combinations(sorted(raters), 2):
        xs = [raters[left].get(i) for i in item_ids]
        ys = [raters[right].get(i) for i in item_ids]
        kappa, agree, n = cohen_kappa(xs, ys)
        lo, hi = wilson(agree, n)
        rows.append({
            "kind": "human",
            "group": "inter_rater",
            "comparison": f"{left}_vs_{right}",
            "n_total": str(n),
            "n_dec": "",
            "a": str(agree),
            "b": str(n - agree),
            "tie": "",
            "rate_pct": f"{100 * agree / n:.1f}" if n else "",
            "ci_low": f"{lo:.1f}" if n else "",
            "ci_high": f"{hi:.1f}" if n else "",
            "p_sign": "",
            "diff_from_50": "",
            "odds_ratio": "",
            "notes": f"cohen_kappa={kappa:.3f}",
        })

    comparisons = sorted({r["comparison"] for r in key.values()})
    for comp in comparisons + ["ALL"]:
        comp_ids = item_ids if comp == "ALL" else [i for i in item_ids if key[i]["comparison"] == comp]
        pooled_human_v: list[str] = []
        pooled_agree = pooled_dec_n = 0
        for rater, choices in sorted(raters.items()):
            human_v, judge_v = [], []
            agree = dec_n = 0
            for item_id in comp_ids:
                h = human_choice_to_system(choices.get(item_id), key[item_id])
                if h is None:
                    continue
                j = key[item_id]["judge_verdict"]
                human_v.append(h)
                judge_v.append(j)
                if h in {"a", "b"} and j in {"a", "b"}:
                    dec_n += 1
                    agree += int(h == j)
                    pooled_dec_n += 1
                    pooled_agree += int(h == j)
            a = human_v.count("a")
            b = human_v.count("b")
            tie = human_v.count("tie")
            pooled_human_v.extend(human_v)
            dec = a + b
            rate = 100 * a / dec if dec else float("nan")
            lo, hi = wilson(a, dec)
            alo, ahi = wilson(agree, dec_n)
            rows.append({
                "kind": "human",
                "group": "human_win_rate",
                "comparison": f"{comp}:{rater}",
                "n_total": str(len(human_v)),
                "n_dec": str(dec),
                "a": str(a),
                "b": str(b),
                "tie": str(tie),
                "rate_pct": f"{rate:.1f}" if not math.isnan(rate) else "",
                "ci_low": f"{lo:.1f}" if not math.isnan(lo) else "",
                "ci_high": f"{hi:.1f}" if not math.isnan(hi) else "",
                "p_sign": fmt_float(sign_p_observed_direction(a, b)),
                "diff_from_50": f"{rate - 50:.1f}" if not math.isnan(rate) else "",
                "odds_ratio": fmt_float(a / b if b else float("inf") if a else float("nan")),
                "notes": f"judge_decisive_agreement={agree}/{dec_n}; agreement_ci={alo:.1f}-{ahi:.1f}",
            })
        a = pooled_human_v.count("a")
        b = pooled_human_v.count("b")
        tie = pooled_human_v.count("tie")
        dec = a + b
        rate = 100 * a / dec if dec else float("nan")
        lo, hi = wilson(a, dec)
        alo, ahi = wilson(pooled_agree, pooled_dec_n)
        rows.append({
            "kind": "human",
            "group": "human_win_rate_pooled",
            "comparison": comp,
            "n_total": str(len(pooled_human_v)),
            "n_dec": str(dec),
            "a": str(a),
            "b": str(b),
            "tie": str(tie),
            "rate_pct": f"{rate:.1f}" if not math.isnan(rate) else "",
            "ci_low": f"{lo:.1f}" if not math.isnan(lo) else "",
            "ci_high": f"{hi:.1f}" if not math.isnan(hi) else "",
            "p_sign": fmt_float(sign_p_observed_direction(a, b)),
            "diff_from_50": f"{rate - 50:.1f}" if not math.isnan(rate) else "",
            "odds_ratio": fmt_float(a / b if b else float("inf") if a else float("nan")),
            "notes": f"judge_decisive_agreement={pooled_agree}/{pooled_dec_n}; agreement_ci={alo:.1f}-{ahi:.1f}",
        })
    return rows


def cross_judge_agreement_rows() -> list[dict[str, str]]:
    pat_map = {
        "TMon_vs_TMoff": "TMon_vs_TMoff",
        "Hygiene_vs_TMon": "Hygiene_vs_TMon",
        "A2_vs_Hygiene": "A2_vs_Hygiene",
        "A2_vs_GG": "A2_vs_GG",
        "RAGBIG_vs_GG": "RAGBIG_vs_GG",
    }
    old_by_pattern: dict[str, dict[tuple[str, str], str]] = {name: {} for name in pat_map}
    for path in JUDGE.glob("pairwise_*deepseek-chat.csv"):
        for name, pat in pat_map.items():
            if pat not in path.name:
                continue
            for r in read_csv(path):
                old_by_pattern[name][(r["chapter"], r["sentence_id"])] = r["final_choice"]

    rows = []
    for path in sorted(JUDGE.glob("rejudge_*_gemini35flash.csv")):
        name = path.stem.replace("rejudge_", "").replace("_gemini35flash", "")
        old = old_by_pattern.get(name, {})
        pairs = []
        for r in read_csv(path):
            key = (r["chapter"], r["sentence_id"])
            if key in old:
                pairs.append((old[key], r["final_choice"]))
        n = len(pairs)
        exact = sum(1 for o, v in pairs if o == v)
        both_dec = [(o, v) for o, v in pairs if o in {"a", "b"} and v in {"a", "b"}]
        dec_agree = sum(1 for o, v in both_dec if o == v)
        dec_n = len(both_dec)
        old_dec = [o for o, _ in pairs if o in {"a", "b"}]
        new_dec = [v for _, v in pairs if v in {"a", "b"}]
        lo, hi = wilson(dec_agree, dec_n)
        rows.append({
            "kind": "validation",
            "group": "cross_judge_agreement",
            "comparison": name,
            "n_total": str(n),
            "n_dec": str(dec_n),
            "a": str(dec_agree),
            "b": str(dec_n - dec_agree),
            "tie": "",
            "rate_pct": f"{100 * dec_agree / dec_n:.1f}" if dec_n else "",
            "ci_low": f"{lo:.1f}" if dec_n else "",
            "ci_high": f"{hi:.1f}" if dec_n else "",
            "p_sign": "",
            "diff_from_50": "",
            "odds_ratio": "",
            "notes": (
                f"exact_agreement={exact}/{n}"
                f"; old_A_rate={100 * old_dec.count('a') / len(old_dec):.1f}"
                f"; new_A_rate={100 * new_dec.count('a') / len(new_dec):.1f}"
            ) if old_dec and new_dec else f"exact_agreement={exact}/{n}",
        })
    return rows


def main() -> None:
    rows: list[dict[str, str]] = []
    chaps_a = [6, 7, 8, 9, 10]
    chaps_b = range(20, 31)

    rows.extend([
        judge_row("TM_on_vs_TM_off", [JUDGE / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"] + chap_paths([7, 8, 9, 10], "TMon_vs_TMoff", "deepseek-chat"), "within_pipeline_A"),
        judge_row("Hygiene_vs_TM_on", chap_paths(chaps_a, "Hygiene_vs_TMon", "deepseek-chat"), "within_pipeline_A"),
        judge_row("A2_vs_Hygiene", chap_paths(chaps_a, "A2_vs_Hygiene", "deepseek-chat"), "within_pipeline_A"),
        judge_row("TM_off_vs_GG", [JUDGE / "pairwise_chap6_TMoff_vs_GG_deepseek-chat.csv"], "vs_GG_A"),
        judge_row("TM_on_vs_GG", chap_paths(chaps_a, "TMon_vs_GG", "deepseek-chat"), "vs_GG_A"),
        judge_row("Hygiene_vs_GG", chap_paths(chaps_a, "Hygiene_vs_GG", "deepseek-chat"), "vs_GG_A"),
        judge_row("A2_vs_GG", chap_paths(chaps_a, "A2_vs_GG", "deepseek-chat"), "vs_GG_A"),
        judge_row("RAG_BIG_vs_GG", chap_paths(chaps_a, "Track3_RAGBIG_vs_GG", "deepseek-chat"), "vs_GG_A"),
        judge_row("RAG_BIG_vs_NORAG_BIG", [JUDGE / "pairwise_chap7_Track3_RAGBIG_vs_NORAGBIG_deepseek-chat.csv"], "scale_70B_A"),
        judge_row("NORAG_BIG_vs_GG", [JUDGE / "pairwise_chap7_Track3_NORAGBIG_vs_GG_deepseek-chat.csv"], "scale_70B_A"),
        judge_row("Qwen_ZS_vs_GG", chap_paths(chaps_b, "QwenZS_vs_GG", "gemini35flash"), "testB"),
        judge_row("Qwen_random_fewshot_vs_GG", chap_paths(chaps_b, "QwenRndFS_vs_GG", "gemini35flash"), "testB"),
        judge_row("Qwen_RAG_A2new_vs_GG", chap_paths(chaps_b, "A2new_vs_GG", "gemini35flash"), "testB"),
        judge_row("DeepSeek_ZS_vs_GG", chap_paths(chaps_b, "DeepSeekZS_vs_GG", "gemini35flash"), "testB"),
        judge_row("DeepSeek_RAG_vs_GG", chap_paths(chaps_b, "DeepSeek_vs_GG", "gemini35flash"), "testB"),
        judge_row("rejudge_TM_on_vs_TM_off", [JUDGE / "rejudge_TMon_vs_TMoff_gemini35flash.csv"], "cross_judge"),
        judge_row("rejudge_Hygiene_vs_TM_on", [JUDGE / "rejudge_Hygiene_vs_TMon_gemini35flash.csv"], "cross_judge"),
        judge_row("rejudge_A2_vs_Hygiene", [JUDGE / "rejudge_A2_vs_Hygiene_gemini35flash.csv"], "cross_judge"),
        judge_row("rejudge_A2_vs_GG", [JUDGE / "rejudge_A2_vs_GG_gemini35flash.csv"], "cross_judge"),
        judge_row("rejudge_RAG_BIG_vs_GG", [JUDGE / "rejudge_RAGBIG_vs_GG_gemini35flash.csv"], "cross_judge"),
    ])

    rows.extend(human_rows())
    rows.extend(cross_judge_agreement_rows())

    fields = ["kind", "group", "comparison", "n_total", "n_dec", "a", "b", "tie",
              "rate_pct", "ci_low", "ci_high", "p_sign", "diff_from_50",
              "odds_ratio", "notes"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"{'group':<18} {'comparison':<30} {'n_dec':>6} {'A%':>6} {'95% CI':>15} {'p':>10}")
    for r in rows:
        if r["kind"] != "judge":
            continue
        ci = f"{r['ci_low']}-{r['ci_high']}" if r["ci_low"] else ""
        print(f"{r['group']:<18} {r['comparison']:<30} {r['n_dec']:>6} {r['rate_pct']:>6} {ci:>15} {r['p_sign']:>10}")


if __name__ == "__main__":
    main()
