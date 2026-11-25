"""Corrected statistics for the LLM-judge pairwise comparisons (reviewer revision).

Adds, for every pairwise judge comparison that ``evaluation/paper_stats.py``
aggregates (``kind == "judge"`` rows):

  * the OLD one-sided exact sign-test p-value in the *observed* direction
    (kept only for side-by-side comparison -- the direction is chosen after
    seeing the data, so it is not a valid p-value);
  * the exact TWO-SIDED binomial p-value (the valid counterpart);
  * a chapter-level cluster bootstrap CI for the decisive win rate, which
    accounts for the fact that sentences are clustered within few chapters
    (the sentence-level Wilson interval is therefore too narrow).

Additionally emits the same statistics with ``scope = chap6_matched`` for the
four cumulative-ablation stages on chapter 6 only, judged by DeepSeek only:
the single fully matched comparison set.

Outputs:
    evaluation/revision_stats_summary.csv

Only the Python standard library is used.  ``paper_stats.py`` is read for its
conventions but is never imported or modified.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
JUDGE = EVAL / "llm_judge"
OUT = EVAL / "revision_stats_summary.csv"

N_BOOT = 10_000
BOOT_SEED = 42
BOOT_LO_Q = 0.025
BOOT_HI_Q = 0.975

FIELDS = [
    "group", "comparison", "scope", "n_total", "n_dec", "a", "b", "tie",
    "rate_pct", "wilson_low", "wilson_high", "p_one_sided_observed",
    "p_two_sided", "cluster_boot_low", "cluster_boot_high", "n_chapters",
    "chapters_present", "notes",
]


# --------------------------------------------------------------------------
# helpers copied verbatim from evaluation/paper_stats.py (conventions)
# --------------------------------------------------------------------------

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


def binom_cdf_le(k: int, n: int) -> float:
    """Exact P(X <= k) for X ~ Binomial(n, 0.5), log-sum-exp stabilised.

    Same formulation as paper_stats.sign_p_observed_direction, so it stays
    numerically stable for n > 1000.
    """
    if n == 0:
        return float("nan")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    return min(math.exp(m) * sum(math.exp(x - m) for x in logs), 1.0)


def sign_p_observed_direction(a: int, b: int) -> float:
    """OLD one-sided exact sign-test p-value in the observed direction."""
    n = a + b
    if n == 0:
        return float("nan")
    return binom_cdf_le(min(a, b), n)


def sign_p_two_sided(a: int, b: int) -> float:
    """Exact two-sided binomial p-value (direction chosen from the data)."""
    n = a + b
    if n == 0:
        return float("nan")
    return min(1.0, 2.0 * binom_cdf_le(min(a, b), n))


def fmt_float(x: float, digits: int = 6) -> str:
    if math.isnan(x):
        return ""
    return f"{x:.{digits}g}"


def fmt_pct(x: float, digits: int = 2) -> str:
    if math.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def chap_paths(chaps: range | list[int], slug: str, judge: str) -> list[Path]:
    return [JUDGE / f"pairwise_chap{c}_{slug}_{judge}.csv" for c in chaps]


def chap_sort_key(chapter: str) -> tuple[int, str]:
    c = str(chapter).strip()
    return (int(c), "") if c.isdigit() else (10 ** 9, c)


def percentile(sorted_vals: list[float], q: float) -> float:
    """Percentile by sorting the replicates and indexing (linear interpolation,
    i.e. position (n-1)*q)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def cluster_bootstrap(
    per_chapter: dict[str, tuple[int, int]],
    rng: random.Random,
) -> tuple[float, float]:
    """Chapter-level cluster bootstrap CI (percent) for the pooled A-rate.

    Resamples the chapters with replacement (all rows of a sampled chapter
    travel together) and recomputes sum(a_i)/sum(a_i+b_i).
    """
    chapters = sorted(per_chapter, key=chap_sort_key)
    n_ch = len(chapters)
    rates: list[float] = []
    for _ in range(N_BOOT):
        sa = sb = 0
        for _ in range(n_ch):
            a_i, b_i = per_chapter[chapters[rng.randrange(n_ch)]]
            sa += a_i
            sb += b_i
        if sa + sb:
            rates.append(100 * sa / (sa + sb))
    if not rates:
        return float("nan"), float("nan")
    rates.sort()
    return percentile(rates, BOOT_LO_Q), percentile(rates, BOOT_HI_Q)


# --------------------------------------------------------------------------
# comparison specs -- copied from evaluation/paper_stats.py main()
# --------------------------------------------------------------------------

CHAPS_A = [6, 7, 8, 9, 10]
CHAPS_B = range(20, 31)


def comparison_specs() -> list[tuple[str, str, list[Path]]]:
    """(comparison, group, expected paths) -- identical to paper_stats.main()."""
    return [
        ("TM_on_vs_TM_off", "within_pipeline_A",
         [JUDGE / "pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv"]
         + chap_paths([7, 8, 9, 10], "TMon_vs_TMoff", "deepseek-chat")),
        ("Hygiene_vs_TM_on", "within_pipeline_A",
         chap_paths(CHAPS_A, "Hygiene_vs_TMon", "deepseek-chat")),
        ("A2_vs_Hygiene", "within_pipeline_A",
         chap_paths(CHAPS_A, "A2_vs_Hygiene", "deepseek-chat")),
        ("TM_off_vs_GG", "vs_GG_A",
         [JUDGE / "pairwise_chap6_TMoff_vs_GG_deepseek-chat.csv"]),
        ("TM_on_vs_GG", "vs_GG_A",
         chap_paths(CHAPS_A, "TMon_vs_GG", "deepseek-chat")),
        ("Hygiene_vs_GG", "vs_GG_A",
         chap_paths(CHAPS_A, "Hygiene_vs_GG", "deepseek-chat")),
        ("A2_vs_GG", "vs_GG_A",
         chap_paths(CHAPS_A, "A2_vs_GG", "deepseek-chat")),
        ("RAG_BIG_vs_GG", "vs_GG_A",
         chap_paths(CHAPS_A, "Track3_RAGBIG_vs_GG", "deepseek-chat")),
        ("RAG_BIG_vs_NORAG_BIG", "scale_70B_A",
         [JUDGE / "pairwise_chap7_Track3_RAGBIG_vs_NORAGBIG_deepseek-chat.csv"]),
        ("NORAG_BIG_vs_GG", "scale_70B_A",
         [JUDGE / "pairwise_chap7_Track3_NORAGBIG_vs_GG_deepseek-chat.csv"]),
        ("Qwen_ZS_vs_GG", "testB",
         chap_paths(CHAPS_B, "QwenZS_vs_GG", "gemini35flash")),
        ("Qwen_random_fewshot_vs_GG", "testB",
         chap_paths(CHAPS_B, "QwenRndFS_vs_GG", "gemini35flash")),
        ("Qwen_RAG_A2new_vs_GG", "testB",
         chap_paths(CHAPS_B, "A2new_vs_GG", "gemini35flash")),
        ("DeepSeek_ZS_vs_GG", "testB",
         chap_paths(CHAPS_B, "DeepSeekZS_vs_GG", "gemini35flash")),
        ("DeepSeek_RAG_vs_GG", "testB",
         chap_paths(CHAPS_B, "DeepSeek_vs_GG", "gemini35flash")),
        ("rejudge_TM_on_vs_TM_off", "cross_judge",
         [JUDGE / "rejudge_TMon_vs_TMoff_gemini35flash.csv"]),
        ("rejudge_Hygiene_vs_TM_on", "cross_judge",
         [JUDGE / "rejudge_Hygiene_vs_TMon_gemini35flash.csv"]),
        ("rejudge_A2_vs_Hygiene", "cross_judge",
         [JUDGE / "rejudge_A2_vs_Hygiene_gemini35flash.csv"]),
        ("rejudge_A2_vs_GG", "cross_judge",
         [JUDGE / "rejudge_A2_vs_GG_gemini35flash.csv"]),
        ("rejudge_RAG_BIG_vs_GG", "cross_judge",
         [JUDGE / "rejudge_RAGBIG_vs_GG_gemini35flash.csv"]),
    ]


# The four cumulative-ablation stages, chapter 6, DeepSeek judge only.
CHAP6_MATCHED = [
    ("TM_off_vs_GG", "vs_GG_A", "pairwise_chap6_TMoff_vs_GG_deepseek-chat.csv"),
    ("TM_on_vs_GG", "vs_GG_A", "pairwise_chap6_TMon_vs_GG_deepseek-chat.csv"),
    ("Hygiene_vs_GG", "vs_GG_A", "pairwise_chap6_Hygiene_vs_GG_deepseek-chat.csv"),
    ("A2_vs_GG", "vs_GG_A", "pairwise_chap6_A2_vs_GG_deepseek-chat.csv"),
]


# --------------------------------------------------------------------------
# statistics row
# --------------------------------------------------------------------------

def stats_row(
    comparison: str,
    group: str,
    scope: str,
    paths: list[Path],
    only_chapter: str | None = None,
    only_judge: str | None = None,
    extra_note: str = "",
) -> dict[str, str]:
    a = b = tie = n_total = 0
    missing: list[str] = []
    per_chapter: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for path in paths:
        if not path.exists():
            missing.append(path.name)
            continue
        for r in read_csv(path):
            if only_chapter is not None and str(r.get("chapter", "")).strip() != only_chapter:
                continue
            if only_judge is not None and only_judge not in (r.get("judge_model") or "").lower():
                continue
            n_total += 1
            choice = r.get("final_choice")
            if choice == "a":
                a += 1
                per_chapter[str(r.get("chapter", "")).strip()][0] += 1
            elif choice == "b":
                b += 1
                per_chapter[str(r.get("chapter", "")).strip()][1] += 1
            elif choice == "tie":
                tie += 1

    n_dec = a + b
    rate = 100 * a / n_dec if n_dec else float("nan")
    w_lo, w_hi = wilson(a, n_dec)

    chapters = sorted(per_chapter, key=chap_sort_key)
    n_chapters = len(chapters)

    notes: list[str] = []
    if missing:
        notes.append("missing_files=" + ",".join(sorted(missing)))

    boot_lo = boot_hi = float("nan")
    if n_chapters >= 2 and n_dec:
        rng = random.Random(BOOT_SEED)
        boot_lo, boot_hi = cluster_bootstrap(
            {c: (per_chapter[c][0], per_chapter[c][1]) for c in chapters}, rng)
    elif n_chapters < 2:
        notes.append(
            f"cluster_bootstrap_omitted: only {n_chapters} chapter(s) present, "
            "cluster bootstrap needs >=2 (single-chapter sampling is degenerate)"
        )
    if n_dec == 0:
        notes.append("no_decisive_verdicts")

    if (not math.isnan(boot_lo) and not math.isnan(boot_hi)
            and not math.isnan(w_lo) and not math.isnan(w_hi)):
        boot_width = boot_hi - boot_lo
        wilson_width = w_hi - w_lo
        notes.append(f"boot_wider_by_pp={boot_width - wilson_width:.2f}")
    if extra_note:
        notes.append(extra_note)

    return {
        "group": group,
        "comparison": comparison,
        "scope": scope,
        "n_total": str(n_total),
        "n_dec": str(n_dec),
        "a": str(a),
        "b": str(b),
        "tie": str(tie),
        "rate_pct": fmt_pct(rate),
        "wilson_low": fmt_pct(w_lo),
        "wilson_high": fmt_pct(w_hi),
        "p_one_sided_observed": fmt_float(sign_p_observed_direction(a, b)),
        "p_two_sided": fmt_float(sign_p_two_sided(a, b)),
        "cluster_boot_low": fmt_pct(boot_lo),
        "cluster_boot_high": fmt_pct(boot_hi),
        "n_chapters": str(n_chapters),
        "chapters_present": ",".join(chapters),
        "notes": "; ".join(notes),
    }


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for comparison, group, paths in comparison_specs():
        rows.append(stats_row(comparison, group, "pooled", paths))
    for comparison, group, name in CHAP6_MATCHED:
        rows.append(stats_row(
            comparison, group, "chap6_matched", [JUDGE / name],
            only_chapter="6", only_judge="deepseek",
            extra_note="chap6_matched: chapter 6 + deepseek-chat judge only",
        ))
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    header = (f"{'group':<17} {'comparison':<26} {'scope':<13} {'n_dec':>6} "
              f"{'rate%':>7} {'wilson 95%':>15} {'p_1s_obs':>11} {'p_2s':>11} "
              f"{'cluster boot 95%':>17} {'nch':>3}")
    print(header)
    print("-" * len(header))
    for r in rows:
        wil = f"{r['wilson_low']}-{r['wilson_high']}" if r["wilson_low"] else ""
        boot = (f"{r['cluster_boot_low']}-{r['cluster_boot_high']}"
                if r["cluster_boot_low"] else "n/a")
        print(f"{r['group']:<17} {r['comparison']:<26} {r['scope']:<13} "
              f"{r['n_dec']:>6} {r['rate_pct']:>7} {wil:>15} "
              f"{r['p_one_sided_observed']:>11} {r['p_two_sided']:>11} "
              f"{boot:>17} {r['n_chapters']:>3}")


def main() -> None:
    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")
    print()
    print_table(rows)

    missing = sorted({m for r in rows for m in r["notes"].split("; ")
                      if m.startswith("missing_files=")})
    print()
    if missing:
        for m in missing:
            print(f"WARNING expected judge CSV missing -> {m}")
    else:
        print("all expected judge CSVs present")


if __name__ == "__main__":
    main()
