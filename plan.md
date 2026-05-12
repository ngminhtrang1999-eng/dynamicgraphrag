# Thesis Plan — Dynamic GraphRAG for Vietnamese Literary MT

**Branch:** `refactor/2026-thesis` · **Last updated:** 2026-05-13

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
| Week 1 validation — human spot-check | **pending USER** (`evaluation/llm_judge/spot_check_for_review.csv`) |  |
| Week 2 — A3 Translation Memory | not started |  |
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

## Next session checklist

1. Read this file and the project memory file (`~/.claude/projects/C--Users-trang/memory/project_dynamic_graphrag.md`).
2. Check `git log` and `git status` on `refactor/2026-thesis` to confirm where work resumed.
3. If user has filled `spot_check_for_review.csv`: run `score_spot_check.py` first.
4. Otherwise: start A3 Translation Memory implementation. Suggested entry point — new module `modules/translation_memory.py` exposing `add(source, translation, score)`, `retrieve(source, k=3)`. Wire into `core/pipeline.py` `PostEditingPipeline._phase_2_draft_and_edit` between Step A (draft) and Step C (edit).
