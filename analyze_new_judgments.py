"""Aggregate new Gemini judgments + judge-agreement analysis.

Usage:
    python analyze_new_judgments.py aggregate   # win rates for all gemini35flash pairwise CSVs
    python analyze_new_judgments.py agreement   # old deepseek vs new gemini verdicts on rejudge sample
"""

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "evaluation" / "llm_judge"
ANALYSIS = Path(__file__).parent / "evaluation" / "analysis"
AGREEMENT_OUTPUT = ANALYSIS / "cross_judge_agreement.csv"
STABILITY_OUTPUT = ANALYSIS / "matched_judge_stability.csv"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
RANDOMIZATION_SAMPLES = 100_000
AGREEMENT_PATTERNS = {
    "A2_vs_GG": "A2_vs_GG",
    "A2_vs_Hygiene": "A2_vs_Hygiene",
    "Hygiene_vs_TMon": "Hygiene_vs_TMon",
    "RAGBIG_vs_GG": "Track3_RAGBIG_vs_GG",
    "TMon_vs_TMoff": "TMon_vs_TMoff",
}


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    fraction = position - lower
    return values[lower] + fraction * (values[min(lower + 1, len(values) - 1)] - values[lower])


def cluster_bootstrap_ci(values, cluster_ids, statistic):
    grouped = defaultdict(list)
    for cluster_id, value in zip(cluster_ids, values):
        grouped[cluster_id].append(value)
    clusters = list(grouped)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [value for cluster in rng.choices(clusters, k=len(clusters))
                  for value in grouped[cluster]]
        estimate = statistic(sample)
        if math.isfinite(estimate):
            estimates.append(estimate)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def cluster_sign_flip_p(differences, cluster_ids):
    grouped = defaultdict(list)
    for cluster_id, difference in zip(cluster_ids, differences):
        grouped[cluster_id].append(difference)
    cluster_sums = [sum(values) for values in grouped.values()]
    observed = abs(sum(cluster_sums))
    if not observed:
        return 1.0
    rng = random.Random(BOOTSTRAP_SEED)
    extreme = sum(abs(sum(total if rng.getrandbits(1) else -total for total in cluster_sums))
                  >= observed for _ in range(RANDOMIZATION_SAMPLES))
    return (extreme + 1) / (RANDOMIZATION_SAMPLES + 1)


def holm_adjust(p_values):
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def stability_summary(family, calls, cluster_ids):
    n = len(calls)
    call1 = [first for first, _, _ in calls]
    call2 = [second for _, second, _ in calls]
    finals = [final for _, _, final in calls]
    exact = sum(first == second for first, second in zip(call1, call2))
    decisive = [(first, second) for first, second in zip(call1, call2)
                if first in {"a", "b"} and second in {"a", "b"}]
    decisive_agreement = sum(first == second for first, second in decisive)
    ab_flips = sum(first != second for first, second in decisive)
    call1_a, call2_a = call1.count("a"), call2.count("a")
    final_ties = finals.count("tie")
    disagreement_ties = sum(final == "tie" and first != second
                             for first, second, final in calls)

    row = {field: "" for field in STABILITY_FIELDS}
    row.update({
        "scope": "judge_family", "judge_family": family, "n_items": n,
        "n_source_clusters": len(set(cluster_ids)),
        "exact_agreement_n": exact, "exact_agreement_denominator": n,
        "decisive_agreement_n": decisive_agreement,
        "decisive_agreement_denominator": len(decisive),
        "ab_flip_n": ab_flips, "ab_flip_denominator": n,
        "call1_system_a_n": call1_a, "call1_system_a_denominator": n,
        "call2_system_a_n": call2_a, "call2_system_a_denominator": n,
        "position_shift_pp": f"{(call2_a - call1_a) / n * 100:.6f}",
        "a_to_non_a_n": sum(first == "a" and second != "a"
                            for first, second in zip(call1, call2)),
        "non_a_to_a_n": sum(first != "a" and second == "a"
                            for first, second in zip(call1, call2)),
        "final_tie_n": final_ties, "final_tie_denominator": n,
        "final_tie_from_disagreement_n": disagreement_ties,
        "final_tie_from_disagreement_denominator": final_ties,
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "randomization_samples": RANDOMIZATION_SAMPLES,
    })
    for prefix, successes, denominator, statistic in (
        ("exact_agreement", exact, n,
         lambda sample: sum(a == b for a, b, _ in sample) / len(sample)),
        ("decisive_agreement", decisive_agreement, len(decisive),
         lambda sample: (sum(a == b for a, b, _ in sample if a in {"a", "b"} and b in {"a", "b"}) /
                         sum(a in {"a", "b"} and b in {"a", "b"} for a, b, _ in sample))),
        ("ab_flip", ab_flips, n,
         lambda sample: sum(a != b and a in {"a", "b"} and b in {"a", "b"}
                            for a, b, _ in sample) / len(sample)),
        ("call1_system_a", call1_a, n,
         lambda sample: sum(a == "a" for a, _, _ in sample) / len(sample)),
        ("call2_system_a", call2_a, n,
         lambda sample: sum(b == "a" for _, b, _ in sample) / len(sample)),
        ("final_tie", final_ties, n,
         lambda sample: sum(final == "tie" for _, _, final in sample) / len(sample)),
        ("final_tie_from_disagreement", disagreement_ties, final_ties,
         lambda sample: (sum(final == "tie" and a != b for a, b, final in sample) /
                         sum(final == "tie" for _, _, final in sample))),
    ):
        low, high = cluster_bootstrap_ci(calls, cluster_ids, statistic)
        row[f"{prefix}_rate"] = f"{successes / denominator:.6f}"
        row[f"{prefix}_ci95_low"] = f"{low:.6f}"
        row[f"{prefix}_ci95_high"] = f"{high:.6f}"
    low, high = cluster_bootstrap_ci(
        calls, cluster_ids,
        lambda sample: sum((b == "a") - (a == "a") for a, b, _ in sample) / len(sample))
    row["position_shift_bootstrap_ci95_low_pp"] = f"{low * 100:.6f}"
    row["position_shift_bootstrap_ci95_high_pp"] = f"{high * 100:.6f}"
    row["discordant_negative_n"] = row["a_to_non_a_n"]
    row["discordant_positive_n"] = row["non_a_to_a_n"]
    differences = [(b == "a") - (a == "a") for a, b, _ in calls]
    row["cluster_randomization_p_raw"] = f"{cluster_sign_flip_p(differences, cluster_ids):.12g}"
    return row


STABILITY_FIELDS = (
    "scope", "judge_family", "n_items", "n_source_clusters",
    "exact_agreement_n", "exact_agreement_denominator", "exact_agreement_rate",
    "exact_agreement_ci95_low", "exact_agreement_ci95_high",
    "decisive_agreement_n", "decisive_agreement_denominator", "decisive_agreement_rate",
    "decisive_agreement_ci95_low", "decisive_agreement_ci95_high",
    "ab_flip_n", "ab_flip_denominator", "ab_flip_rate", "ab_flip_ci95_low", "ab_flip_ci95_high",
    "call1_system_a_n", "call1_system_a_denominator", "call1_system_a_rate",
    "call1_system_a_ci95_low", "call1_system_a_ci95_high",
    "call2_system_a_n", "call2_system_a_denominator", "call2_system_a_rate",
    "call2_system_a_ci95_low", "call2_system_a_ci95_high",
    "position_shift_pp", "position_shift_bootstrap_ci95_low_pp",
    "position_shift_bootstrap_ci95_high_pp", "a_to_non_a_n", "non_a_to_a_n",
    "final_tie_n", "final_tie_denominator", "final_tie_rate", "final_tie_ci95_low",
    "final_tie_ci95_high", "final_tie_from_disagreement_n",
    "final_tie_from_disagreement_denominator", "final_tie_from_disagreement_rate",
    "final_tie_from_disagreement_ci95_low", "final_tie_from_disagreement_ci95_high",
    "deepseek_stable_n", "gemini_stable_n", "gemini_only_stable_n", "deepseek_only_stable_n",
    "stability_rate_gap_pp", "stability_gap_bootstrap_ci95_low_pp",
    "stability_gap_bootstrap_ci95_high_pp", "discordant_negative_n",
    "discordant_positive_n", "cluster_randomization_p_raw", "cluster_randomization_holm_p",
    "bootstrap_samples", "bootstrap_seed", "randomization_samples",
)


def aggregate():
    groups = defaultdict(list)
    for p in sorted(OUT.glob("pairwise_chap*_gemini35flash.csv")):
        # pairwise_chap<N>_<SLUG>_vs_GG_gemini35flash.csv
        slug = p.stem.split("_", 2)[2].rsplit("_gemini35flash", 1)[0]
        groups[slug].extend(read(p))
    print(f"{'comparison':<28} {'n':>5} {'n_dec':>6} {'A%':>7}")
    for slug, rows in sorted(groups.items()):
        a = sum(1 for r in rows if r["final_choice"] == "a")
        b = sum(1 for r in rows if r["final_choice"] == "b")
        n_dec = a + b
        pct = 100 * a / n_dec if n_dec else float("nan")
        print(f"{slug:<28} {len(rows):>5} {n_dec:>6} {pct:>6.1f}%")


def agreement():
    new_paths = sorted(OUT.glob("rejudge_*_gemini35flash.csv"))
    found_names = {path.stem.removeprefix("rejudge_").removesuffix("_gemini35flash")
                   for path in new_paths}
    if found_names != set(AGREEMENT_PATTERNS):
        raise ValueError(f"expected rejudge files for {sorted(AGREEMENT_PATTERNS)}; got {sorted(found_names)}")

    print(f"{'comparison':<18} {'exact (95% CI)':>25} {'decisive (95% CI)':>29} "
          f"{'oldA%':>7} {'newA%':>7}")
    output_rows = []
    matched_records = []
    total_pairs = 0
    for newp in new_paths:
        name = newp.stem.replace("rejudge_", "").replace("_gemini35flash", "")
        new_rows = read(newp)
        new_keys = [(row["chapter"], row["sentence_id"]) for row in new_rows]
        if len(new_rows) != 100 or len(set(new_keys)) != 100:
            raise ValueError(f"{newp.name}: expected 100 unique (chapter, sentence_id) items")
        new = {}
        for key, row in zip(new_keys, new_rows):
            calls = tuple((row.get(column) or "").strip().lower()
                          for column in ("call_1_choice", "call_2_choice_unswapped", "final_choice"))
            invalid = set(calls) - {"a", "b", "tie"}
            if invalid:
                raise ValueError(f"{newp.name}: invalid choices {sorted(invalid)}")
            new[key] = calls

        old = {}
        pattern = AGREEMENT_PATTERNS[name]
        for oldp in sorted(OUT.glob(f"pairwise_*{pattern}*deepseek-chat.csv")):
            for r in read(oldp):
                key = (r["chapter"], r["sentence_id"])
                if key in old:
                    raise ValueError(f"{name}: duplicate old item {key}")
                calls = tuple((r.get(column) or "").strip().lower()
                              for column in ("call_1_choice", "call_2_choice_unswapped", "final_choice"))
                invalid = set(calls) - {"a", "b", "tie"}
                if invalid:
                    raise ValueError(f"{oldp.name}: invalid choices {sorted(invalid)}")
                old[key] = calls
        matched_keys = [key for key in new if key in old]
        pairs = [(old[key][2], new[key][2]) for key in matched_keys]
        n = len(pairs)
        if n != 100:
            raise ValueError(f"{name}: expected 100 matched pairs; got {n}")
        matched_records.extend((name, key, old[key], new[key]) for key in matched_keys)
        total_pairs += n
        exact_n = sum(o == v for o, v in pairs)
        exact_ci = cluster_bootstrap_ci(
            pairs, matched_keys, lambda sample: sum(a == b for a, b in sample) / len(sample))
        both_dec = [(o, v) for o, v in pairs if o in ("a", "b") and v in ("a", "b")]
        decisive_n = sum(o == v for o, v in both_dec)
        decisive_total = len(both_dec)
        decisive_ci = cluster_bootstrap_ci(
            pairs, matched_keys,
            lambda sample: (sum(a == b for a, b in sample if a in {"a", "b"} and b in {"a", "b"}) /
                            sum(a in {"a", "b"} and b in {"a", "b"} for a, b in sample)))
        old_a = [o for o, _ in pairs if o in ("a", "b")]
        new_a = [v for _, v in pairs if v in ("a", "b")]
        old_pct = 100 * old_a.count("a") / len(old_a) if old_a else float("nan")
        new_pct = 100 * new_a.count("a") / len(new_a) if new_a else float("nan")
        output_rows.append({
            "comparison": name, "matched_n": n, "n_source_clusters": len(set(matched_keys)),
            "exact_agreement_n": exact_n, "exact_agreement_denominator": n,
            "exact_agreement_rate": f"{exact_n / n:.6f}",
            "exact_agreement_ci95_low": f"{exact_ci[0]:.6f}",
            "exact_agreement_ci95_high": f"{exact_ci[1]:.6f}",
            "decisive_agreement_n": decisive_n,
            "decisive_agreement_denominator": decisive_total,
            "decisive_agreement_rate": f"{decisive_n / decisive_total:.6f}",
            "decisive_agreement_ci95_low": f"{decisive_ci[0]:.6f}",
            "decisive_agreement_ci95_high": f"{decisive_ci[1]:.6f}",
            "old_candidate_wins": old_a.count("a"), "old_decisive_n": len(old_a),
            "old_candidate_rate": f"{old_pct / 100:.6f}",
            "new_candidate_wins": new_a.count("a"), "new_decisive_n": len(new_a),
            "new_candidate_rate": f"{new_pct / 100:.6f}",
        })
        print(f"{name:<18} {exact_n:>3}/{n:<3}={exact_n / n:>5.1%} "
              f"({exact_ci[0]:.1%}-{exact_ci[1]:.1%}) "
              f"{decisive_n:>3}/{decisive_total:<3}={decisive_n / decisive_total:>5.1%} "
              f"({decisive_ci[0]:.1%}-{decisive_ci[1]:.1%}) "
              f"{old_pct:>6.1f}% {new_pct:>6.1f}%")

    if total_pairs != 500:
        raise ValueError(f"expected 500 total matched pairs; got {total_pairs}")
    item_keys = {(name, *key) for name, key, _, _ in matched_records}
    if len(matched_records) != 500 or len(item_keys) != 500:
        raise ValueError("expected 500 unique (comparison, chapter, sentence_id) items per judge")
    cluster_ids = [key for _, key, _, _ in matched_records]
    if len(set(cluster_ids)) != 340:
        raise ValueError(f"expected 340 unique source sentences; got {len(set(cluster_ids))}")

    deepseek_calls = [old for _, _, old, _ in matched_records]
    gemini_calls = [new for _, _, _, new in matched_records]
    deepseek_row = stability_summary("deepseek-chat", deepseek_calls, cluster_ids)
    gemini_row = stability_summary("gemini35flash", gemini_calls, cluster_ids)
    deepseek_stable = [first == second for first, second, _ in deepseek_calls]
    gemini_stable = [first == second for first, second, _ in gemini_calls]
    gemini_only = sum(gemini and not deepseek
                      for deepseek, gemini in zip(deepseek_stable, gemini_stable))
    deepseek_only = sum(deepseek and not gemini
                        for deepseek, gemini in zip(deepseek_stable, gemini_stable))
    stability_pairs = list(zip(deepseek_stable, gemini_stable))
    low, high = cluster_bootstrap_ci(
        stability_pairs, cluster_ids,
        lambda sample: sum(gemini - deepseek for deepseek, gemini in sample) / len(sample))
    cross_p = cluster_sign_flip_p(
        [gemini - deepseek for deepseek, gemini in stability_pairs], cluster_ids)
    cross_row = {field: "" for field in STABILITY_FIELDS}
    cross_row.update({
        "scope": "cross_family", "judge_family": "gemini35flash_minus_deepseek-chat",
        "n_items": total_pairs, "n_source_clusters": len(set(cluster_ids)),
        "deepseek_stable_n": sum(deepseek_stable), "gemini_stable_n": sum(gemini_stable),
        "gemini_only_stable_n": gemini_only, "deepseek_only_stable_n": deepseek_only,
        "stability_rate_gap_pp": f"{(sum(gemini_stable) - sum(deepseek_stable)) / total_pairs * 100:.6f}",
        "stability_gap_bootstrap_ci95_low_pp": f"{low * 100:.6f}",
        "stability_gap_bootstrap_ci95_high_pp": f"{high * 100:.6f}",
        "discordant_negative_n": deepseek_only, "discordant_positive_n": gemini_only,
        "cluster_randomization_p_raw": f"{cross_p:.12g}",
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "randomization_samples": RANDOMIZATION_SAMPLES,
    })
    stability_rows = [deepseek_row, gemini_row, cross_row]
    raw_p = [float(row["cluster_randomization_p_raw"]) for row in stability_rows]
    for row, adjusted in zip(stability_rows, holm_adjust(raw_p)):
        row["cluster_randomization_holm_p"] = f"{adjusted:.12g}"

    reproduced = {
        "deepseek": (deepseek_row["exact_agreement_n"], deepseek_row["call1_system_a_n"],
                     deepseek_row["call2_system_a_n"], deepseek_row["a_to_non_a_n"],
                     deepseek_row["non_a_to_a_n"]),
        "gemini": (gemini_row["exact_agreement_n"], gemini_row["call1_system_a_n"],
                   gemini_row["call2_system_a_n"], gemini_row["a_to_non_a_n"],
                   gemini_row["non_a_to_a_n"]),
        "cross": (gemini_only, deepseek_only),
    }
    expected = {"deepseek": (347, 293, 161, 142, 10),
                "gemini": (450, 234, 222, 31, 19), "cross": (130, 27)}
    if reproduced != expected:
        raise ValueError(f"matched-500 reproduction failed: expected {expected}; got {reproduced}")

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(AGREEMENT_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with open(STABILITY_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STABILITY_FIELDS)
        writer.writeheader()
        writer.writerows(stability_rows)
    print(f"Validated {total_pairs} comparison-items from {len(set(cluster_ids))} source sentences")
    print(f"Wrote {AGREEMENT_OUTPUT}")
    print(f"Wrote {STABILITY_OUTPUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    {"aggregate": aggregate, "agreement": agreement}[mode]()
