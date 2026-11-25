# Human evaluation kit (expanded, blind, multi-rater)

Addresses the reviewer requirement to grow human corroboration from n=30 (one
rater, chapter 6) to a large blind set with **chance-corrected inter-rater
agreement (Cohen's / Fleiss' κ)** and a **directly paper-comparable human
win-rate**.

## Files

| File | Role |
|------|------|
| `human_eval_pairs.csv` | **Blind sheet.** 501 items. System identity hidden; A/B order randomized. *Not in the public repository (contains novel text); rebuild with `python build_human_eval_kit.py`.* |
| `human_eval_key.csv` | **Truth (do not show raters).** True A/B identity + the LLM judge's verdict per item. IDs and labels only; published. |
| `rater_app.html` | **One-click rating app** for non-technical raters (offline, double-click to open; exports `human_eval_rater_<name>.csv`). Regenerate with `build_rater_html.py`. *Not published (embeds the text).* |
| `human_eval_rater_<name>.csv` | A rater's completed copy (from the app, or a hand-filled sheet copy). *Not published (private research records).* |

## Design

501 items (~167 each), 3 headline comparisons from **test set B (chapters 20–30)**:

- `deepseek_rag_vs_gg` — DeepSeek-V3.1 + RAG vs Google Translate
- `qwen_a2_vs_gg` — Qwen3-8B A2 full RAG vs Google Translate
- `deepseek_zs_vs_gg` — DeepSeek-V3.1 zero-shot vs Google Translate

**Representative sampling.** Within each comparison, items are drawn from the
Gemini judge's three verdict buckets (decisive-A / decisive-B / tie) **in
proportion to their pool sizes**, so the sampled judge decisive-win rate
approximates the corresponding headline numbers. Human rates and human--judge
agreement are therefore reported as **complementary corroboration**, not as
validation or calibration: sampling was conditioned on Gemini verdicts and the
human rubric differs from the LLM rubric. Reproducible: seed 42.

The 501 comparison-items contain 436 unique `(chapter, sentence_id)` sources.
Reference text is available for 364 items and absent for 137; raters use it only
when present. Aggregate uncertainty must resample by source identity.

## Rating criteria (priority high → low)

Raters pick the better translation by these criteria, applied top-down:

1. **Semantic completeness.** A translation may be shorter or merge clauses, but
   must convey the full meaning of the source. Missing or wrong meaning loses outright.
2. **Character address & consistency.** Pronouns / forms of address between
   characters (anh/em/ngài/tiểu thư…) must fit the relationship and stay
   consistent throughout.
3. **No foreign-script leakage.** No stray Chinese/English tokens mid-sentence;
   proper names correctly romanized to Sino-Vietnamese (e.g. "Chu Minh Thụy",
   not "Zhou Mingrui").

If both are equal on all three, pick the more fluent/natural reading; still
equal → tie. (These factors are shown to every rater in `rater_app.html`.)

## How to run a rating round

**Easiest (non-technical raters):** send each rater the single file
`rater_app.html`. They double-click it, type a name, judge all items with mouse
or keyboard (1=A, 2=tie, 3=B), and download `human_eval_rater_<name>.csv` to
send back. No install, no internet. Progress autosaves in their browser.

**Manual alternative:**

1. Give each rater a copy of `human_eval_pairs.csv` renamed
   `human_eval_rater_<name>.csv`. Recommend ≥2 raters (≥3 enables Fleiss' κ).
2. Each rater reads `source` (+ `reference` if present) and the two
   translations `candidate_A` / `candidate_B`, then writes `A`, `B`, or `tie`
     in the **`choice`** column. They must not see `human_eval_key.csv`.

3. Put completed files back in this folder and run, from the repo root:

   ```bash
   python score_human_eval.py
   ```

## Output of the scorer

- Pairwise **Cohen's κ** for every rater pair (+ raw % agreement).
- **Fleiss' κ** across all raters when ≥3 are present.
- Per comparison: human decisive-win rate vs the LLM judge's, plus per-item
  human–judge decisive agreement.

Rebuild the sheet (seed 42, reproducible) with `python build_human_eval_kit.py`.

## Participation record

Both raters gave informed consent and were compensated. Public reporting uses
the aliases **Rater 1** and **Rater 2**; raw filenames and annotations remain
private research records. No institutional-review approval is claimed here.
