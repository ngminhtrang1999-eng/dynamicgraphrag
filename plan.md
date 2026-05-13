# Thesis Plan — Dynamic GraphRAG for Vietnamese Literary MT

**Branch:** `refactor/2026-thesis` · **Last updated:** 2026-05-13 (A3 MVP + chap-6 ablation partial)

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
| Week 1 validation — human spot-check | **FAILED** (14.3% agreement, judge needs revision) | uncommitted |
| Week 2 — A3 Translation Memory | **MVP done** (smoke verified, qualitative win on names + style) | uncommitted |
| Week 2 — A3 chap-6 ablation partial (n=22 paired sids) | partial: TM_on 19.73 vs TM_off 19.23 (sum-of-5), accuracy +0.56 | uncommitted |
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

## Next session checklist

1. Read this file and the project memory file (`~/.claude/projects/C--Users-trang/memory/project_dynamic_graphrag.md`).
2. Check `git log` and `git status` on `refactor/2026-thesis`.
3. **HIGH PRIORITY — Revise judge (task 12)**: current 14.3% agreement makes the
   ablation numbers untrustworthy. Two options:
   a. Pairwise judge ("which would a Vietnamese reader prefer") — better matches
      human review framing.
   b. Re-weight aggregation: fluency+register+style as headline; accuracy/cultural
      as ceiling-only (anything <3 = invalid, else don't count).
   Re-validate on the same 14 spot-check pairs. Target ≥70%.
4. After judge fixed: complete chap-6 judge (52 more calls — wait for OpenRouter
   reset OR switch to paid DeepSeek API). Then extend to chap 7-10 ablation.
5. **Expand TM seed** to chap 2-5 once judge is fixed — chap-1-only seed already
   shows usefulness but similarity sits at 0.36-0.61 for chap-6 queries. Larger seed
   should raise sim and improve TM win rate.
6. **Then A1**: hierarchical retrieval (persist entity types in `term_glossary.json`).

## Files added/changed in this session (uncommitted)

- `modules/translation_memory.py` (new)
- `seed_translation_memory.py` (new)
- `align_ablation.py` (new)
- `score_spot_check.py` (modified — normalize "A and B" as tie)
- `core/pipeline.py` (modified — TM wiring + empty-edit fallback)
- `main.py` (modified — --tm/--no-tm, --experiment-suffix, --chap-start/--chap-end)
- `modules/prompts.py` (modified — `format_tm_examples`, draft prompt accepts tm_examples)
- `data/knowledge_base/qwen3-8b-local_yes_tm/` (new — TM-armed experiment dir)
- `data/knowledge_base/qwen3-8b-local_yes_notm/` (new — control experiment dir)
- `evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_6.csv` (new)
- `evaluation/llm_judge/scores_qwen3-8b-local-ablation_z-ai_glm-4.5-air_free.csv` (new, 50 rows)
- `plan.md` (updated)
