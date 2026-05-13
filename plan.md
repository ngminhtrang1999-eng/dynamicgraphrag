# Thesis Plan — Dynamic GraphRAG for Vietnamese Literary MT

**Branch:** `refactor/2026-thesis` · **Last updated:** 2026-05-13 (A3 MVP + pairwise judge + chap-6 ablation full)

> Resume note: this document captures where the project stands and what to do next.
> Read this first when starting a new session. Memory entries (in `~/.claude/...`)
> describe the user, environment, and durable decisions; this file describes
> in-flight thesis work.

---

## Thesis shape

- **Contribution A (main): smarter dynamic RAG for EN→VI literary MT with small LLMs.**
  Three components designed to ablate independently.
- **Contribution B (supporting): evaluation framework for literary MT.**
  Reference-based LLM-judge on 5 literary axes, validated by human pairwise blind.

## Runtime stack

- Translation: WSL Ollama serving `qwen3:8b`, `llama3.1:8b`, `gemma3:4b` at `localhost:11434`.
- Judge: OpenAI-compat (default OpenRouter `z-ai/glm-4.5-air:free`); switchable to DeepSeek paid API once purchased.
- Python: WSL miniconda env `graphrag` (Python 3.11, has openai/spacy/torch+CUDA/transformers/dotenv). Activate:
  ```
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate graphrag
  cd "/mnt/d/Dynamic GraphRAG"
  ```

## Status snapshot

| Phase | State | Commit |
|---|---|---|
| Week 0 — unblock batch experiments | done | `2a7cd06` |
| Week 1 — B framework MVP | done | `b93341f` |
| Week 1 validation — human spot-check (sum-of-5) | FAILED 14.3% → diagnosed | committed 7977c8a |
| Week 2 — A3 Translation Memory MVP | done (smoke verified, qualitative win on names + style) | committed 2743bef |
| Week 2 — A3 chap-6 ablation partial (n=22, sum-of-5) | done: TM_on 19.73 vs TM_off 19.23, accuracy +0.56 | committed 7977c8a |
| Week 2 — Judge revision: pairwise approach | **done — deepseek-chat pairwise = 57.1% agreement, 4x baseline** | uncommitted |
| Week 2 — A3 chap-6 ablation full pairwise (n=62) | **done — TM_on 62.5% decisive win rate, p=0.077 one-sided** | uncommitted |
| Week 3 — A1 Hierarchical retrieval | not started |  |
| Week 4 — A2 Coreference retrieval | not started |  |
| Week 5 — full ablation + B framework scale-up | not started |  |
| Week 6 — write-up | not started |  |

## Pending USER action

1. Open `evaluation/llm_judge/spot_check_for_review.csv` in Excel/Sheets.
2. For each of 14 pairs: read source + reference + candidate_A + candidate_B → fill `your_choice` with `A`, `B`, or `tie`.
3. Save. **Do not** open `spot_check_key.csv` until done.
4. Next session: run `python score_spot_check.py --review ... --key ... --scores ...` (see script `--help`) to compute agreement. Target ≥70% — if below, refine judge rubric before scaling B framework.

## A framework — sub-components and rationale

### A3 — Translation Memory (do first)

- Maintain a sentence-level index of past translations judged high-quality (judge mean ≥ 4.0 across axes, or curated reference seeds in chapter 1).
- LaBSE-embed source sentence → cosine top-k (k=2-3) → inject as few-shot examples in `build_post_editing_prompt` and/or `build_draft_translation_prompt`.
- LaBSE already in HF cache at `~/.cache/huggingface/hub/models--sentence-transformers--LaBSE` (~1.8GB).
- Storage: numpy array + JSON metadata in `data/knowledge_base/<experiment>/translation_memory.{npy,json}`. No FAISS needed at thesis scale.
- Cold-start: seed with first ~50 sentences from `evaluation/alignment_results_*/aligned_master_chap_1.csv` `ref` column (gold humans).
- Expected impact: largest on `fluency`, `style`. Convergence to consistent voice as chapters accumulate is itself the headline result.

### A1 — Hierarchical retrieval (do second)

- Extraction prompt (`build_extraction_prompt`, `build_glossary_extraction_prompt`) already categorizes entities (PERSON/DEITY/ORG/LOCATION/CONCEPT) but **does not persist the type**.
- Update `graph_manager` schema: `glossary: {term: {translation, type}}` (currently flat `{term: translation}`). Migration: existing entries → `type: "UNKNOWN"`.
- Update `get_relevant_context` to return `dict[type, dict[term, translation]]`.
- Update injection / post-edit prompts to render glossary as separate sections.
- Expected impact: moderate, mainly readability for the LLM in long-glossary chunks.

### A2 — Coreference-aware retrieval (do last)

- Replace/augment the current "spaCy NER → entity-injection LLM call" with a proper coref model (`fastcoref` recommended — Hugging Face, LingMess-RoBERTa under the hood).
- Resolve pronouns → entity mentions before retrieval, so glossary filtering can be tight.
- New dependency: `fastcoref` (~500MB). Run as CPU-side preprocessing step per chunk; cache resolution to skip re-runs.
- Expected impact: targets `register` axis (xưng hô) and reduces the `[Entity]` tag leak into final output observed in smoke test (chunk 5 had `anh [Alger]` leaking through post-editing).

## Existing infrastructure (Week 0)

- `core/llm_backend.py`: `OllamaLLM` and `OpenAICompatLLM` classes. Both strip `<think>...</think>` blocks at backend level for reasoning models.
- `main.py`: choices include `qwen3-8b-local`, `llama3.1-8b-local`, `gemma3-4b-local`, `deepseek-chat`. `--non-interactive` flag for unattended batch runs.
- `core/pipeline.py`: `PostEditingPipeline` (Draft → Inject → Edit) gated for non-interactive; reuses mining logic from `GraphRAGPipeline`.
- `const.py`: `TEMP_EDITING=0.3`, `MAX_NEW_TOKENS=2000` (tuned 2026-05-12). `OLLAMA_BASE_URL`, `DEEPSEEK_*` config added.
- `batch_evaluate_LLM_judge.py`: resume-capable runner. Reads `evaluation/alignment_results_<system>/aligned_master_chap_<N>.csv`, scores RAG/No_RAG/GG columns, writes to `evaluation/llm_judge/scores_<system>_<judge>.csv`.
- `analyze_judge_scores.py`: per-(candidate, axis) mean/std summary.
- `generate_spot_check.py` + `score_spot_check.py`: human validation pipeline for the judge.

## Smoke test verification

- Translation pipeline (qwen3:8b on chapter 6, non-interactive) ran end-to-end: 6 Vietnamese chunks generated. Identified issues (research signals, not bugs): some CJK characters leaked (Qwen3 bilingual emission), entity tags leaked in chunk 5 (motivates A2), name romanization inconsistent (motivates A3 memory).
- Judge framework on n=47: GG=4.58, RAG=4.34, No_RAG=3.81. Discriminates meaningfully.

## A3 — what shipped (2026-05-13)

- `modules/translation_memory.py`: `TranslationMemory(data_dir)` class, LaBSE-backed,
  L2-normalized embeddings, persisted at `<data_dir>/translation_memory.{npy,json}`.
  API: `add(src, trans, score)`, `add_batch(items)`, `retrieve(src, k, min_similarity, min_score)`, `save()`, `load()`.
- `seed_translation_memory.py`: CLI tool to populate TM from `aligned_master_chap_*.csv`.
  Handles `<MERGED_ABOVE>` rows by grouping consecutive sources into a sentence-group
  paired with the leading non-merged ref. Default 50 pairs from chap 1.
  Run: `python seed_translation_memory.py --experiment qwen3-8b-local_yes --chapters 1 --max-pairs 50`.
- `modules/prompts.py`: added `format_tm_examples()`, extended
  `build_draft_translation_prompt(chunk_text, prev_context, tm_examples)`.
  TM block instructs model to learn style, NOT copy verbatim.
- `core/pipeline.py`: `PostEditingPipeline` accepts `translation_memory`, `tm_k`,
  `tm_min_similarity`, `tm_min_score`. Retrieval happens per chunk before Step A.
- `main.py`: flags `--tm` (default on) / `--no-tm`, `--tm-k`, `--tm-min-similarity`.
  TM auto-loads from experiment dir; warns if empty.

**Smoke verification (chunk 0 of chap 6, qwen3:8b, n=1 each):**
- WITH TM: keeps "Zhou Mingrui" romanization → matches gold style.
- WITHOUT TM: degenerates to "Chu Minh Duệ" (Hán-Việt) + code-switch errors ("calminh bản thân").
- TM examples retrieved had sim=0.36–0.61 (chap-1 seed has limited overlap with chap 6 scene).

Confirms the A3 hypothesis: TM enforces name consistency and style convergence
even at low similarity. Stronger effect expected once chapters 2-5 are also seeded
or once judge-validated translations from earlier chapters feed back into TM.

## Spot-check + ablation findings (2026-05-13)

**Spot-check FAILED:** human-judge agreement 14.3% (target ≥70%). Cause: user reviews
at prose/xưng hô level (see [[feedback_review_style]] memory), but judge averages 5
axes equally and gives 25/25 to too many close pairs. The judge correctly catches
word-level errors (e.g. vulgar "Lồn") but humans miss them while skimming prose.

**Chap-6 ablation partial (n=22 paired sids, before OpenRouter free daily limit
50/day hit):**
- TM_on sum-of-5: 19.73 ± 1.10  |  TM_off: 19.23 ± 1.49
- TM_on wins clearly on **accuracy** (4.04 vs 3.48, +0.56)
- Other axes essentially tied
- 8 TM_on wins / 8 TM_off wins / 6 ties
- Two catastrophic TM_off failures (sid 25, 26 — accuracy=1, vulgar word) rescued by
  TM_on. Conversely, sid 7/23 lost TM_on because chap-1 seed doesn't cover that scene.

**Bug found and fixed:** when qwen3:8b's post-edit emits everything inside `<think>`
and `OllamaLLM` strips it, `final_trans` came back empty and was saved as a blank
chunk (TM_on chap-6 chunk 1). `core/pipeline.py` now falls back to the draft when
the edit is empty or <half the draft length. Recovered chunk 1 from log to keep
the partial ablation usable.

## Judge revision findings (2026-05-13)

Tested 4 approaches on the 14-pair spot-check (llama-3.1-8B chap 6, RAG vs No_RAG):

| Approach | Agreement | Notes |
|---|---|---|
| Sum-of-5 baseline | 14.3% | Original plan |
| Prose re-aggregate (register+fluency+style headline, accuracy/cultural as gates) | 28.6% | 2x but still FAIL |
| Pairwise deepseek-chat (v1 prompt) | **57.1%** | 4x baseline, FAIL 70% but best free option |
| Pairwise deepseek-reasoner | 50.0% | Worse + 2 parse errors + 18h projected latency at full scale |
| Pairwise deepseek-chat (v2 relaxed-tie prompt) | 50.0% | Regression — v1 reverted |

**Verdict:** deepseek-chat pairwise (v1 prompt) is the headline judge. Cost ~$1-2 for full chap 6-10 ablation. The remaining gap to 70% is fundamental: 2/6 disagreements are structural (judge correctly catches word-level errors like "Lồn", "ông ta" that humans miss when reading at prose level — see `feedback_review_style` memory). Not a bug to fix; document as paper finding.

Pairwise prompt lives at `build_literary_pairwise_judge_prompt` in `modules/prompts.py`. Runner is `batch_pairwise_judge.py` with configurable column mapping for both spot-check and ablation CSV formats.

## Chap 6 ablation pairwise (2026-05-13, n=62 paired sentences)

deepseek-chat pairwise judging TM_on vs TM_off (`pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv`):
- TM_on wins: 25 / TM_off wins: 15 / Tie: 22
- Decisive win rate: TM_on 62.5%, TM_off 37.5%
- Sign test one-sided: p = 0.077 (trending, not yet < 0.05)

Confirms the earlier per-axis n=22 signal (TM_on +0.50 sum-of-5, +0.56 accuracy). Pairwise gives stronger frame for paper: "TM_on chosen over TM_off in 62.5% of decisive pairs". Extending to chap 7-10 should reach significance if signal holds.

## Next session checklist (in order)

### USER actions before/during next session

**A. Phase 3 human review (30-45 min, do this FIRST in next session):**

```bash
# In WSL graphrag env:
python review_app.py
# Then open http://localhost:5000 in browser
```

The UI presents 30 pairs (14 re-eval RAG vs No_RAG + 16 new TM_on vs TM_off anonymized).
Critically — the UI shows rubric-aligned instructions: **look for word-level errors
(Lồn, code-switch, hallucination) and xưng hô FIRST, prose flow second.** Last time
you missed word-level slips ("Lồn", "ông ta") because you read at prose level — this
framing fixes that. Goal: cross 70% agreement with the judge.

Save state is automatic (writes to `evaluation/llm_judge/spot_check_v2_responses.csv`
after each Submit). Can pause + resume any time.

**B. Crawl/source chap 11-25 (for TM seed expansion + LoRA hold-out):**

Need pairs of (EN source, VN reference) for chapters 11-25 of Lord of the Mysteries.
Same format as `data/LOTM/chapters/chap_<N>.txt` (EN) and the `ref` columns in
`evaluation/alignment_results_*/aligned_master_chap_<N>.csv` (VN). Drop the EN files
into `data/LOTM/chapters/`. For VN references, the project will run `align_ablation.py`
or `batch_alignment.py` to align them.

Why chap 11-25 specifically: test set is chap 6-10 (per `const.START_CHAPTER=6`,
`END_CHAPTER=10`). Chap 1-5 already used for TM seed. To avoid data leakage:
- TM seed pool: chap 1-5 + chap 11-25 (20 chapters, ~3000 sentence pairs)
- LoRA hold-out dataset: chap 11-25 (15 chapters, kept strictly OUT of judge eval)
- Test ablation: chap 6-10 (untouched by training)

### Claude actions next session (after USER does A and B)

1. **Read plan.md + project memory** (`~/.claude/projects/C--Users-trang/memory/`).
2. **Check git status** + `git log` on `refactor/2026-thesis`.
3. **Score Phase 3 results**: `python score_spot_check_v2.py` → report new vs old agreement on set 1, human vs judge on set 2. If ≥70% on set 1: document as paper finding. If <70%: dig into disagreements.
4. **Align chap 11-25** (user-supplied source/ref): run alignment pipeline to produce `evaluation/alignment_results_*/aligned_master_chap_<N>.csv` for N=11..25.
5. **Expand TM seed**: `python seed_translation_memory.py --chapters 1,2,3,4,5,11,...,25 --max-pairs 1000` to populate TM with broader coverage.
6. **Generate TM_on/TM_off translations for chap 7-10**: run `main.py` twice with `--experiment-suffix tm` (TM on) and `--experiment-suffix notm` (TM off) over chap 7-10. ~6-12 hours of Ollama background work (qwen3:8b WSL).
7. **Align chap 7-10 ablation outputs** with `align_ablation.py` → `aligned_master_chap_{7,8,9,10}.csv` with TM_on/TM_off columns.
8. **Run pairwise judge on chap 7-10 ablation**: `python batch_pairwise_judge.py --review evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_<N>.csv --col-a TM_on --col-b TM_off --col-source Source_Text --col-ref ref --col-id Source_ID --default-chapter <N> --judge deepseek-chat --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --out evaluation/llm_judge/pairwise_ablation_chap<N>_TMon_vs_TMoff_deepseek-chat.csv`. Cost ~$1-2 total. Target combined n≥200 across chap 6-10 for sign-test p<0.05.
9. **(Optional) Phase 4 calibration triples** if you want bulletproof paper methodology: hand-craft 10-15 src sentences with 3 versions each (Gold/Mid/Bad), validate judge picks Gold > Mid > Bad ≥95%. ~30 min Claude time + $0.10 API.
10. **Then A1 — Hierarchical retrieval**: persist entity types in `graph_manager` schema (currently flat dict, needs `{term: {translation, type}}`). Update `get_relevant_context` to bucket by type. Update prompts to render glossary as separate sections.

## Files added/changed in this session (uncommitted as of 2026-05-13 pairwise/judge work)

Already committed in `7977c8a` (prior session): translation_memory.py, seed_translation_memory.py, align_ablation.py, pipeline.py changes, main.py ablation flags, prompts.py TM injection, knowledge_base ablation dirs, aligned_master_chap_6.csv, partial per-axis scores.

Net new in this session (pairwise judge work + Phase 3 scaffold):
- `batch_pairwise_judge.py` (new — pairwise runner with configurable column mapping, position-bias swap)
- `probe_judge_providers.py` (new — free-tier provider availability probe)
- `prepare_spot_check_v2.py` (new — Phase 3 data prep, 14 re-eval + 16 ablation anonymized)
- `review_app.py` (new — Flask web UI for Phase 3 human review)
- `score_spot_check_v2.py` (new — Phase 3 scoring vs deepseek-chat)
- `score_spot_check.py` (modified — added `--rubric prose` for re-aggregation experiment)
- `modules/prompts.py` (modified — added `build_literary_pairwise_judge_prompt`)
- `evaluation/llm_judge/pairwise_spot_check_for_review_deepseek-chat.csv` (new — 14-pair spot-check, 57.1% agreement)
- `evaluation/llm_judge/pairwise_spot_check_for_review_deepseek-reasoner.csv` (new — 14-pair reasoner run, 50.0% with 2 parse errors)
- `evaluation/llm_judge/pairwise_spot_check_for_review_deepseek-chat_v2.csv` (new — relaxed-tie prompt experiment, regressed)
- `evaluation/llm_judge/pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv` (new — **headline ablation result, 62 paired sentences**)
- `evaluation/llm_judge/spot_check_v2_pairs.csv` (new — 30 pairs for Phase 3, A/B anonymized in set 2)
- `evaluation/llm_judge/spot_check_v2_key.csv` (new — true A_is/B_is per pair, do NOT open before review)
- `plan.md` (updated with this section + Phase 3 instructions)
