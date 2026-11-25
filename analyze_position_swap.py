"""Analyze position-swap consistency for the 93 main-paper two-call CSVs."""

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
JUDGMENTS = ROOT / "evaluation" / "llm_judge"
OUTPUT = ROOT / "evaluation" / "analysis" / "position_swap_summary.csv"
CHOICES = {"a", "b", "tie"}
CHOICE_COLUMNS = {"call_1_choice", "call_2_choice_unswapped", "final_choice"}
REQUIRED_COLUMNS = {"chapter", "sentence_id"} | CHOICE_COLUMNS
EXPECTED_FILES = 93
# The immutable CSVs currently contain 10,033 paired rows; 12,357 is not
# reproducible from this 93-file manifest.
EXPECTED_ROWS = 10_033
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def chapter_files(chapters, comparison, judge):
    return tuple(f"pairwise_chap{chapter}_{comparison}_{judge}.csv" for chapter in chapters)


# Explicit experiment manifest: no directory-wide glob, pilots, or spot checks.
MANIFEST = (
    ("TMoff_vs_GG", "deepseek-chat", ("pairwise_chap6_TMoff_vs_GG_deepseek-chat.csv",)),
    ("TMon_vs_GG", "deepseek-chat", chapter_files(range(6, 11), "TMon_vs_GG", "deepseek-chat")),
    ("Hygiene_vs_GG", "deepseek-chat", chapter_files(range(6, 11), "Hygiene_vs_GG", "deepseek-chat")),
    ("A2_vs_GG", "deepseek-chat", chapter_files(range(6, 11), "A2_vs_GG", "deepseek-chat")),
    ("TMon_vs_TMoff", "deepseek-chat", ("pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv",)
     + chapter_files(range(7, 11), "TMon_vs_TMoff", "deepseek-chat")),
    ("Hygiene_vs_TMon", "deepseek-chat", chapter_files(range(6, 11), "Hygiene_vs_TMon", "deepseek-chat")),
    ("A2_vs_Hygiene", "deepseek-chat", chapter_files(range(6, 11), "A2_vs_Hygiene", "deepseek-chat")),
    ("RAGBIG_vs_GG", "deepseek-chat", chapter_files(range(6, 11), "Track3_RAGBIG_vs_GG", "deepseek-chat")),
    ("NORAGBIG_vs_GG", "deepseek-chat", ("pairwise_chap7_Track3_NORAGBIG_vs_GG_deepseek-chat.csv",)),
    ("RAGBIG_vs_NORAGBIG", "deepseek-chat", ("pairwise_chap7_Track3_RAGBIG_vs_NORAGBIG_deepseek-chat.csv",)),
    ("A2new_vs_GG", "gemini35flash", chapter_files(range(20, 31), "A2new_vs_GG", "gemini35flash")),
    ("DeepSeekZS_vs_GG", "gemini35flash", chapter_files(range(20, 31), "DeepSeekZS_vs_GG", "gemini35flash")),
    ("DeepSeek_vs_GG", "gemini35flash", chapter_files(range(20, 31), "DeepSeek_vs_GG", "gemini35flash")),
    ("QwenRndFS_vs_GG", "gemini35flash", chapter_files(range(20, 31), "QwenRndFS_vs_GG", "gemini35flash")),
    ("QwenZS_vs_GG", "gemini35flash", chapter_files(range(20, 31), "QwenZS_vs_GG", "gemini35flash")),
)


def rate(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    fraction = position - lower
    return values[lower] + fraction * (values[min(lower + 1, len(values) - 1)] - values[lower])


def cluster_bootstrap_ci(records, value):
    grouped = defaultdict(lambda: [0, 0])
    for _, _, _, row in records:
        summary = grouped[(row["chapter"], row["sentence_id"])]
        summary[0] += value(row)
        summary[1] += 1
    clusters = list(grouped.values())
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.choices(clusters, k=len(clusters))
        estimates.append(sum(item[0] for item in sample) / sum(item[1] for item in sample))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize(scope, group, records):
    rows = [row for _, _, _, row in records]
    files = {filename for _, _, filename, _ in records}
    n = len(rows)
    exact_n = sum(row["call_1_choice"] == row["call_2_choice_unswapped"] for row in rows)
    both_decisive = [row for row in rows if row["call_1_choice"] in {"a", "b"}
                     and row["call_2_choice_unswapped"] in {"a", "b"}]
    decisive_agree_n = sum(row["call_1_choice"] == row["call_2_choice_unswapped"]
                           for row in both_decisive)
    flip_n = sum(row["call_1_choice"] != row["call_2_choice_unswapped"]
                 for row in both_decisive)
    call1_a = sum(row["call_1_choice"] == "a" for row in rows)
    call2_a = sum(row["call_2_choice_unswapped"] == "a" for row in rows)
    final_ties = [row for row in rows if row["final_choice"] == "tie"]
    disagreement_ties = sum(row["call_1_choice"] != row["call_2_choice_unswapped"]
                            for row in final_ties)
    family = next(iter({family for _, family, _, _ in records})) if scope == "comparison" else group
    if scope == "overall":
        family = "all"
    source_clusters = {(row["chapter"], row["sentence_id"]) for row in rows}
    summary = {
        "scope": scope, "group": group, "judge_family": family,
        "file_count": len(files), "n_items": n, "n_source_clusters": len(source_clusters),
        "exact_agreement_n": exact_n, "exact_agreement_denominator": n,
        "exact_agreement_rate": f"{rate(exact_n, n):.6f}",
        "decisive_agreement_n": decisive_agree_n,
        "decisive_agreement_denominator": len(both_decisive),
        "decisive_agreement_rate": f"{rate(decisive_agree_n, len(both_decisive)):.6f}",
        "ab_flip_n": flip_n, "ab_flip_denominator": n,
        "ab_flip_rate": f"{rate(flip_n, n):.6f}",
        "call1_system_a_n": call1_a, "call1_system_a_denominator": n,
        "call1_system_a_rate": f"{rate(call1_a, n):.6f}",
        "call2_system_a_n": call2_a, "call2_system_a_denominator": n,
        "call2_system_a_rate": f"{rate(call2_a, n):.6f}",
        "call2_minus_call1_pp": f"{(rate(call2_a, n) - rate(call1_a, n)) * 100:.6f}",
        "final_tie_n": len(final_ties),
        "final_tie_from_disagreement_n": disagreement_ties,
        "final_tie_from_disagreement_denominator": len(final_ties),
        "final_tie_from_disagreement_rate": f"{rate(disagreement_ties, len(final_ties)):.6f}",
        "exact_agreement_ci95_low": "", "exact_agreement_ci95_high": "",
        "position_shift_ci95_low_pp": "", "position_shift_ci95_high_pp": "",
        "bootstrap_samples": "", "bootstrap_seed": "",
    }
    if scope == "judge_family":
        exact_ci = cluster_bootstrap_ci(
            records, lambda row: row["call_1_choice"] == row["call_2_choice_unswapped"])
        shift_ci = cluster_bootstrap_ci(
            records, lambda row: ((row["call_2_choice_unswapped"] == "a") -
                                  (row["call_1_choice"] == "a")))
        summary.update({
            "exact_agreement_ci95_low": f"{exact_ci[0]:.6f}",
            "exact_agreement_ci95_high": f"{exact_ci[1]:.6f}",
            "position_shift_ci95_low_pp": f"{shift_ci[0] * 100:.6f}",
            "position_shift_ci95_high_pp": f"{shift_ci[1] * 100:.6f}",
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        })
    return summary


def main():
    filenames = [filename for _, _, files in MANIFEST for filename in files]
    if len(MANIFEST) != 15 or len(filenames) != EXPECTED_FILES or len(set(filenames)) != EXPECTED_FILES:
        raise ValueError("manifest must name 15 comparisons and 93 unique files")

    records = []
    missing = []
    for comparison, family, files in MANIFEST:
        for filename in files:
            path = JUDGMENTS / filename
            if not path.exists():
                missing.append(filename)
                continue
            with open(path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not REQUIRED_COLUMNS <= set(reader.fieldnames or ()):
                    raise ValueError(f"{filename}: missing required choice columns")
                for line, row in enumerate(reader, 2):
                    for column in CHOICE_COLUMNS:
                        row[column] = (row[column] or "").strip().lower()
                        if row[column] not in CHOICES:
                            raise ValueError(f"{filename}:{line}: invalid {column}={row[column]!r}")
                    records.append((comparison, family, filename, row))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} manifest files: {', '.join(missing)}")
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} rows in the 93-file manifest; got {len(records):,}")

    by_comparison, by_family = defaultdict(list), defaultdict(list)
    for record in records:
        by_comparison[record[0]].append(record)
        by_family[record[1]].append(record)
    summaries = [summarize("comparison", name, by_comparison[name]) for name, _, _ in MANIFEST]
    summaries += [summarize("judge_family", family, rows) for family, rows in sorted(by_family.items())]
    summaries.append(summarize("overall", "all", records))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Validated {EXPECTED_FILES} files, {len(records):,} two-call rows, choices in {sorted(CHOICES)}")
    print(f"{'scope/group':<34} {'n':>6} {'exact':>8} {'decisive':>10} {'flip':>8} {'A2-A1':>8} {'discord ties':>14}")
    for row in summaries:
        label = f"{row['scope']}/{row['group']}"
        print(f"{label:<34} {row['n_items']:>6} {float(row['exact_agreement_rate']):>7.1%} "
              f"{float(row['decisive_agreement_rate']):>9.1%} {float(row['ab_flip_rate']):>7.1%} "
              f"{float(row['call2_minus_call1_pp']):>+7.1f}pp "
              f"{row['final_tie_from_disagreement_n']:>5}/{row['final_tie_n']:<5}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
