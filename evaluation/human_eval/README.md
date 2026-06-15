# Human evaluation kit (expanded, blind, multi-rater)

Addresses the reviewer requirement to grow human validation from n=30 (one
rater, chapter 6) to **n≥100 with multiple raters and chance-corrected
agreement (Cohen's / Fleiss' κ)**.

## Files

| File | Role |
|------|------|
| `human_eval_pairs.csv` | **Blind sheet.** 120 items. System identity hidden; A/B order randomized. This is what each rater fills. |
| `human_eval_key.csv` | **Truth (do not show raters).** True A/B identity + the LLM judge's verdict per item. |
| `human_eval_rater_<name>.csv` | A rater's completed copy (you create these). |

## Design

120 items, 3 headline comparisons from **test set B (chapters 20–30)**, 40 each:

- `deepseek_rag_vs_gg` — DeepSeek-V3.1 + RAG vs Google Translate
- `qwen_a2_vs_gg` — Qwen3-8B A2 full RAG vs Google Translate
- `deepseek_zs_vs_gg` — DeepSeek-V3.1 zero-shot vs Google Translate

Each comparison is **stratified by the Gemini judge's verdict** (16 decisive-A,
16 decisive-B, 8 tie) so the sheet spans the full agreement space. Because the
sample is verdict-balanced by design, the validation targets are **per-item
human–judge agreement and inter-rater κ**, not the aggregate win-rate (which is
~50% on this balanced subset by construction).

## How to run a rating round

1. Give each rater a copy of `human_eval_pairs.csv` renamed
   `human_eval_rater_<name>.csv` (e.g. `human_eval_rater_trang.csv`).
   Recommend ≥2 raters (≥3 enables Fleiss' κ).
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
