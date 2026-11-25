"""Score the blind human evaluation without pooling raters as independent items."""

import csv
import glob
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
HE = ROOT / "evaluation" / "human_eval"
ANALYSIS = ROOT / "evaluation" / "analysis"
KEY = HE / "human_eval_key.csv"
LABELS = ["A", "B", "tie"]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
CORROBORATION_OUTPUT = ANALYSIS / "human_judge_corroboration.csv"


def norm(choice):
    c = (choice or "").strip().lower()
    return {"a": "A", "b": "B", "tie": "tie"}.get(c)


def cohen_kappa(labels_a, labels_b):
    pairs = [(x, y) for x, y in zip(labels_a, labels_b) if x and y]
    n = len(pairs)
    if not n:
        return float("nan"), 0
    observed = sum(x == y for x, y in pairs) / n
    cnt_a, cnt_b = defaultdict(int), defaultdict(int)
    for x, y in pairs:
        cnt_a[x] += 1
        cnt_b[y] += 1
    expected = sum(cnt_a[label] * cnt_b[label] / n**2 for label in LABELS)
    return ((observed - expected) / (1 - expected) if expected != 1 else float("nan")), n


def wilson(successes, n, z=1.959963984540054):
    if not n:
        return float("nan"), float("nan")
    p = successes / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return center - margin, center + margin


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    fraction = pos - lo
    return values[lo] + fraction * (values[min(lo + 1, len(values) - 1)] - values[lo])


def decisive_counts(verdicts):
    a = verdicts.count("a")
    b = verdicts.count("b")
    return a, a + b


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


def decisive_gap(pairs):
    human_a, human_n = decisive_counts([human for human, _ in pairs])
    judge_a, judge_n = decisive_counts([judge for _, judge in pairs])
    return human_a / human_n - judge_a / judge_n if human_n and judge_n else float("nan")


def load_key():
    with open(KEY, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row["item_id"] for row in rows]
    if len(rows) != 501 or len(set(ids)) != len(ids):
        raise ValueError(f"human key must contain 501 unique item_id values; got {len(rows)} rows")
    invalid = {row["judge_verdict"] for row in rows} - {"a", "b", "tie"}
    if invalid:
        raise ValueError(f"invalid judge verdicts: {sorted(invalid)}")
    return {row["item_id"]: row for row in rows}


def load_raters(key_ids):
    raters = {}
    for path in sorted(glob.glob(str(HE / "human_eval_rater_*.csv"))):
        name = Path(path).stem.removeprefix("human_eval_rater_")
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        ids = [row["item_id"] for row in rows]
        choices = [norm(row.get("choice")) for row in rows]
        if not any(choices):
            print(f"  (skipping {name}: no choices filled)")
            continue
        if len(rows) != 501 or len(set(ids)) != len(ids) or set(ids) != key_ids:
            raise ValueError(f"rater {name} must contain each of the 501 key items exactly once")
        if not all(choices):
            raise ValueError(f"rater {name} contains blank or invalid choices")
        raters[name] = dict(zip(ids, choices))
        print(f"  loaded completed rater file: {len(rows)} choices")
    return raters


def human_choice_to_system(choice, keyrow):
    if choice == "tie":
        return "tie"
    side_is = keyrow["A_is"] if choice == "A" else keyrow["B_is"]
    return "a" if side_is == "CAND" else "b"


def write_csv(path, rows):
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not KEY.exists():
        raise SystemExit(f"missing {KEY}; run build_human_eval_kit.py first")
    key = load_key()
    print("Loading raters...")
    raters = load_raters(set(key))
    if not raters:
        raise SystemExit("no rater files with filled choices found")

    names = sorted(raters)
    rater_labels = {name: f"Rater {index}" for index, name in enumerate(names, 1)}
    item_ids = sorted(key, key=int)
    cluster_ids = [(key[item_id]["chapter"], key[item_id]["sentence_id"]) for item_id in item_ids]
    if len(set(cluster_ids)) != 436:
        raise ValueError(f"expected 436 unique source sentences; got {len(set(cluster_ids))}")
    comparisons = sorted({key[item_id]["comparison"] for item_id in item_ids})

    print("\n=== Inter-rater agreement (all items) ===")
    interrater_rows = []
    for rater_a, rater_b in itertools.combinations(names, 2):
        labels_a = [raters[rater_a][item_id] for item_id in item_ids]
        labels_b = [raters[rater_b][item_id] for item_id in item_ids]
        kappa, n = cohen_kappa(labels_a, labels_b)
        agreement_n = sum(a == b for a, b in zip(labels_a, labels_b))
        values = list(zip(labels_a, labels_b))
        ci_low, ci_high = cluster_bootstrap_ci(
            values, cluster_ids, lambda sample: sum(a == b for a, b in sample) / len(sample))
        kappa_ci = cluster_bootstrap_ci(
            values, cluster_ids,
            lambda sample: cohen_kappa([a for a, _ in sample], [b for _, b in sample])[0])
        interrater_rows.append({
            "rater_a": rater_labels[rater_a], "rater_b": rater_labels[rater_b],
            "n_items": n, "n_source_clusters": len(set(cluster_ids)),
            "cohen_kappa": f"{kappa:.6f}", "raw_agreement_n": agreement_n,
            "cohen_kappa_ci95_low": f"{kappa_ci[0]:.6f}",
            "cohen_kappa_ci95_high": f"{kappa_ci[1]:.6f}",
            "raw_agreement_rate": f"{agreement_n / n:.6f}",
            "raw_agreement_ci95_low": f"{ci_low:.6f}",
            "raw_agreement_ci95_high": f"{ci_high:.6f}",
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        })
        print(f"  Cohen kappa {rater_labels[rater_a]} vs {rater_labels[rater_b]}: {kappa:.3f} "
              f"(cluster 95% CI {kappa_ci[0]:.3f}-{kappa_ci[1]:.3f}; "
              f"raw agree {agreement_n}/{n} = {agreement_n / n:.1%}, "
              f"cluster 95% CI {ci_low:.1%}-{ci_high:.1%})")
    if not interrater_rows:
        print("  only one rater -- no inter-rater agreement to compute")
    else:
        write_csv(ANALYSIS / "human_interrater_summary.csv", interrater_rows)

    print("\n=== Human vs Gemini-family judge corroboration (candidate vs GG) ===")
    header = (f"{'comparison':<22} {'rater':<10} {'human A%':>9} {'judge A%':>9} "
              f"{'gap (bootstrap 95% CI)':>29} {'decisive agree (95% CI)':>28}")
    print(header)
    print("-" * len(header))
    corroboration_rows = []
    rates_by_source = {rater_labels[name]: {} for name in names}
    rates_by_source["judge"] = {}
    for comparison in comparisons:
        comp_items = [item_id for item_id in item_ids if key[item_id]["comparison"] == comparison]
        comp_cluster_ids = [(key[item_id]["chapter"], key[item_id]["sentence_id"])
                            for item_id in comp_items]
        system = key[comp_items[0]]["candidate_system"]
        judge_verdicts = [key[item_id]["judge_verdict"] for item_id in comp_items]
        judge_a, judge_n = decisive_counts(judge_verdicts)
        judge_rate = judge_a / judge_n
        judge_ci = wilson(judge_a, judge_n)
        rates_by_source["judge"][system] = judge_rate
        for name in names:
            pairs = [(human_choice_to_system(raters[name][item_id], key[item_id]),
                      key[item_id]["judge_verdict"]) for item_id in comp_items]
            human_a, human_n = decisive_counts([human for human, _ in pairs])
            human_rate = human_a / human_n
            human_ci = wilson(human_a, human_n)
            gap = human_rate - judge_rate
            gap_ci = cluster_bootstrap_ci(pairs, comp_cluster_ids, decisive_gap)
            decisive_pairs = [(human, judge) for human, judge in pairs
                              if human in {"a", "b"} and judge in {"a", "b"}]
            agree_n = sum(human == judge for human, judge in decisive_pairs)
            agree_total = len(decisive_pairs)
            agree_ci = wilson(agree_n, agree_total)
            rates_by_source[rater_labels[name]][system] = human_rate
            corroboration_rows.append({
                "comparison": comparison, "candidate_system": system,
                "rater": rater_labels[name], "n_items": len(pairs),
                "n_source_clusters": len(set(comp_cluster_ids)), "human_candidate_wins": human_a,
                "human_decisive_n": human_n, "human_rate": f"{human_rate:.6f}",
                "human_rate_ci95_low": f"{human_ci[0]:.6f}",
                "human_rate_ci95_high": f"{human_ci[1]:.6f}",
                "judge_candidate_wins": judge_a, "judge_decisive_n": judge_n,
                "judge_rate": f"{judge_rate:.6f}",
                "judge_rate_ci95_low": f"{judge_ci[0]:.6f}",
                "judge_rate_ci95_high": f"{judge_ci[1]:.6f}",
                "human_judge_gap_pp": f"{gap * 100:.6f}",
                "gap_bootstrap_ci95_low_pp": f"{gap_ci[0] * 100:.6f}",
                "gap_bootstrap_ci95_high_pp": f"{gap_ci[1] * 100:.6f}",
                "decisive_agreement_n": agree_n, "decisive_agreement_denominator": agree_total,
                "decisive_agreement_rate": f"{agree_n / agree_total:.6f}",
                "decisive_agreement_ci95_low": f"{agree_ci[0]:.6f}",
                "decisive_agreement_ci95_high": f"{agree_ci[1]:.6f}",
            })
            print(f"{comparison:<22} {rater_labels[name]:<10} {human_rate:>8.1%} {judge_rate:>8.1%} "
                  f"{gap * 100:>+6.1f}pp ({gap_ci[0] * 100:+.1f}, {gap_ci[1] * 100:+.1f}) "
                  f"{agree_n:>3}/{agree_total:<3}="
                  f"{agree_n / agree_total:>5.1%} ({agree_ci[0]:.1%}-{agree_ci[1]:.1%})")
    write_csv(CORROBORATION_OUTPUT, corroboration_rows)

    print("\n=== Three-system rank consistency ===")
    judge_order = [system for system, _ in sorted(rates_by_source["judge"].items(),
                                                   key=lambda item: item[1], reverse=True)]
    for source, system_rates in rates_by_source.items():
        ordered = sorted(system_rates.items(), key=lambda item: item[1], reverse=True)
        suffix = "" if source == "judge" else (" [matches judge]" if [x[0] for x in ordered] == judge_order
                                                  else " [differs from judge]")
        print(f"  {source}: " + " > ".join(f"{system} ({rate:.1%})" for system, rate in ordered) + suffix)

    print("\nThese separate-rater results provide aggregate-level corroboration only; "
          "per-item agreement and rank consistency are reported explicitly.")
    print(f"Wrote {ANALYSIS / 'human_interrater_summary.csv'}")
    print(f"Wrote {CORROBORATION_OUTPUT}")


if __name__ == "__main__":
    main()
