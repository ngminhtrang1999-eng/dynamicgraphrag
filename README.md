# Dynamic RAG for English→Vietnamese literary translation

Mã nguồn thực nghiệm cho luận văn thạc sĩ *Dynamic RAG cho dịch văn học Anh–Việt* (Nguyễn Hữu Minh Trạng, 2026).
Experimental code for the master's thesis on retrieval-augmented, post-editing LLM translation of a web novel
(*Lord of the Mysteries*) from English into Vietnamese, with pairwise LLM-judge and human evaluation.

> The repository name `dynamicgraphrag` is historical. The pipeline that the thesis evaluates is the
> draft → post-edit pipeline in `core/pipeline.py` with a typed glossary, translation memory (TM) and
> coreference (A2) components; the early knowledge-graph prototype lives in `stable/` and `old/` (legacy).

**No novel text is distributed.** The novel, its official English text and all Vietnamese translations are
copyrighted. This repository contains code, aggregate numbers and **ID + label** files only. Section
[Rebuilding the text](#rebuilding-the-text) explains how to recreate the text columns from your own copy.

## Contents

| Path | What it is |
|------|------------|
| `main.py`, `core/`, `modules/`, `const.py`, `utils.py` | Translation pipeline (glossary extraction, draft, post-edit, TM, coreference). All prompts are in `modules/prompts.py`. |
| `crawl_docln.py`, `crawl_vntq.py`, `extract_pdf_chapters.py`, `pdf_text_repair.py`, `pdf_split_chapters.py`, `_glyph_inventory.py`, `preprocess_data.py` | Data collection and cleaning of the Vietnamese reference translations and English chapters. |
| `align_reference.py`, `align_ablation.py`, `batch_alignment.py`, `seed_translation_memory.py` | Sentence alignment (LaBSE) and translation-memory seeding. |
| `batch_pairwise_judge.py`, `batch_judge_gemini.py`, `build_rejudge_sample.py`, `sample_deepseek_for_claude_judge.py` | Two-call (A/B-swapped) pairwise LLM judge. |
| `batch_evaluate_*.py` | Automatic metrics (chrF++, COMET, COMET-QE, XCOMET, BLEURT, entity). |
| `analyze_*.py`, `score_*.py`, `evaluation/*.py`, `evaluation/analysis/*.py` | Statistics behind every results table. |
| `build_human_eval_kit.py`, `build_rater_html.py`, `score_human_eval.py` | Blind human-evaluation kit and scoring. |
| `_run_*.sh`, `_align_*.sh`, `_judge_*.sh` | The exact command sequences used for each experiment (run from the repository root under WSL/Linux). |
| `labels/llm_judge/` | Published LLM-judge verdicts, ID + label columns only. |
| `evaluation/` | Aggregate results (`*_summary.csv`, `analysis/*.csv`, `objective_metrics/*/_summary.csv`) and the human-eval key. |
| `thesis/thesis-template-clean/tools/dataset_stats.py` | Dataset statistics reported in Chapter 3. |
| `test_*.py`, `probe_judge_providers.py`, `_debug_*`, `_smoke_*` | API smoke tests and one-off probes; not needed for reproduction. |

## Installation

```bash
conda create -n graphrag python=3.11 -y
conda activate graphrag
pip install -r requirements.txt
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
cp .env.example .env          # then fill in the API keys you need
```

Local models were served with [Ollama](https://ollama.com) 0.23 (`qwen3:8b`, `llama3.1:8b`, `gemma3:4b`).
The thesis runs used WSL2 Ubuntu 24.04 with an RTX 4070 Laptop GPU (8 GB).

## Rebuilding the text

Everything that contains novel text is git-ignored (`data/`, `evaluation/alignment_results_*`,
per-sentence `aligned_master_chap_*.csv`, judge rationales). To rebuild it:

1. Obtain the English chapters and the Vietnamese translations from their original sources yourself and place
   them under `data/LOTM/` (`chapters/` for English, `chapters_vi*/` and `translations/reference/` for Vietnamese).
   `preprocess_data.py` splits the English text into chapters.
2. Vietnamese references: `crawl_docln.py`, `crawl_vntq.py`, or your own copy of the PDF with
   `extract_pdf_chapters.py --pdf <file>` → `pdf_text_repair.py` → `pdf_split_chapters.py`.
   The crawlers are provided **for research use only**; respect each site's terms. vietnamthuquan
   has prohibited crawling since 10 June 2026, so `crawl_vntq.py` is kept only as documentation of how the data was collected.
3. Align: `align_reference.py`, then `align_ablation.py` / `batch_alignment.py` (see `_align_*.sh`).
   This recreates `evaluation/alignment_results_*/aligned_master_chap_N.csv`, keyed by `(chapter, Source_ID)`.
4. Join the text back onto the published labels with the key `(chapter, sentence_id)`.

LLM translation outputs are not bit-reproducible (draft temperature 1.3, no fixed seed, retired
`deepseek-chat` endpoint). Every step **from the published verdicts to the tables** is deterministic.

## Reproducing the thesis tables and figures

Restore the published verdicts first (this never overwrites existing files):

```bash
python scripts/restore_labels.py
```

| Thesis table / figure (LaTeX label) | Content | Command | Input → output |
|---|---|---|---|
| `tab:ch3-crawl`, `tab:data-pdf`, `tab:ch3-ref-sources`, `tab:data-ngram`, `tab:ch3-stats`, `tab:data-per-chapter`, `tab:data-align`, `tab:data-tm`, `tab:data-cosine`, `tab:data-glossary` | Dataset statistics (needs rebuilt text) | `python thesis/thesis-template-clean/tools/dataset_stats.py [--labse]` | `data/LOTM/…`, alignment CSVs → `dataset_stats_output.md` |
| `fig:ch3-judge-evolution` | Judge development | `generate_spot_check.py`, `score_spot_check.py`, `score_spot_check_v2.py` | `labels/llm_judge/spot_check_*` |
| `tab:ch4-runs` | Run matrix | `main.py` via `_run_*.sh` | → `data/LOTM/translations/trans_chap_N_<config>.txt` |
| `tab:ketqua_so_sanh`, `tab:ch4-wlt`, `tab:thesis-test-a`, `tab:thesis-test-b` | Ablations and comparison with Google Translate (win/loss/tie, Wilson CI, sign test) | `python evaluation/paper_stats.py`, `python evaluation/revision_stats.py` | `evaluation/llm_judge/pairwise_chap*.csv` → `evaluation/paper_stats_summary.csv`, `revision_stats_summary.csv` |
| `tab:thesis-test-a/b` (metric columns), `fig:ch4-metrics`, `fig:ch4-chrf-chapters`, `fig:ch4-metric-judge` | chrF++ / COMET | `batch_evaluate_metrics.py` (needs rebuilt text) | → `evaluation/objective_metrics/*/_summary.csv` (published) |
| `tab:cross-judge-rates`, `tab:appendix-rejudge` | Cross-judge agreement, re-judge sample | `python analyze_new_judgments.py` | `pairwise_*`, `rejudge_*` → `evaluation/analysis/cross_judge_agreement.csv`, `matched_judge_stability.csv` |
| `fig:position-bias` | Position bias of the two-call judge | `python analyze_position_swap.py` | → `evaluation/analysis/position_swap_summary.csv` |
| `tab:thesis-human` | Human evaluation | `python score_human_eval.py` (kit: `build_human_eval_kit.py`) | `evaluation/human_eval/` key + rater labels → `evaluation/analysis/human_*.csv` |
| `fig:per-chapter`, `tab:appendix-chap-a`, `tab:appendix-chap-b` | Per-chapter rates and chapter features | `python evaluation/per_chapter_analysis.py` | → `evaluation/analysis/per_chapter_rates.csv`; `per_chapter_features.csv` needs rebuilt text |
| `tab:ch4-examples`, `tab:ch4-themes` | Qualitative examples, rationale themes | `evaluation/analysis/extract_examples.py`, `mine_rationales*.py` | need full judge outputs with rationales (regenerate by re-judging) |
| `tab:pilot-auto` | Pilot automatic metrics | `batch_evaluate_COMET.py`, `batch_evaluate_BLEURT.py`, `batch_evaluate_COMET_QE.py`, `batch_evaluate_entity_overall.py` | → `evaluation/*_overall_results.csv`, `evaluation/{comet,comet_qe,bleurt-20,m-eta}/` (published) |
| `tab:pilot-chapter-judge` | Pilot chapter-level judge | `evaluate_LLM.py`, `batch_evaluate_LLM_runner.py` | outputs quote the novel and are not published |

Configuration tables (`tab:ch4-passes`, `tab:ch4-hparams`, `tab:method-ablation`, `tab:temperatures`) take
their values directly from `const.py`, `core/pipeline.py` and `main.py`.

## Published data format

`labels/llm_judge/pairwise_chap<N>_<candidate>_vs_<baseline>_<judge>.csv`

| Column | Meaning |
|---|---|
| `chapter`, `sentence_id` | Key into `aligned_master_chap_N.csv` (`Source_ID`) |
| `call_1_choice` | Verdict with candidate in position A (`a` / `b` / `tie`) |
| `call_2_choice_unswapped` | Verdict of the A/B-swapped call, mapped back to the original labels |
| `final_choice` | `a` or `b` only if both calls agree, otherwise `tie` |
| `judge_model` | Judge model id |

Removed columns: `rationale_1`, `rationale_2`, and any `source`, `reference`, `candidate_*`, `notes`.

## Human evaluation

| File | Content |
|---|---|
| `evaluation/human_eval/human_eval_key.csv` | For each of the 501 items: comparison, `(chapter, sentence_id)`, which system was shown as A/B, and the LLM judge's verdict |
| `evaluation/human_eval/human_eval_rater_rater1.csv`, `…_rater2.csv` | Each rater's choice per item (`item_id`, `choice` ∈ {`A`, `B`, `tie`}); raters are identified by alias only |

`python score_human_eval.py` and `python evaluation/paper_stats.py` reproduce Cohen's κ and the human vs
judge rates (`evaluation/analysis/human_*.csv`) from these files alone. The blind sheet with the text
(`human_eval_pairs.csv`) is not distributed; rebuild it with `python build_human_eval_kit.py` (seed 42)
once the aligned CSVs exist.

## License and citation

Code: MIT (see `LICENSE`). The ID+label files under `labels/` and `evaluation/` are released under the same terms.
The license does not cover the novel *Lord of the Mysteries*, its translations, or any text derived from them;
none of that text is distributed here.

```bibtex
@misc{nguyen2026code,
  author       = {Nguyễn Hữu Minh Trạng},
  title        = {Mã nguồn thực nghiệm: Dynamic RAG cho dịch văn học Anh--Việt},
  howpublished = {\url{https://github.com/ngminhtrang1999-eng/dynamicgraphrag}},
  note         = {Phiên bản v1.0-thesis},
  year         = {2026}
}
```
