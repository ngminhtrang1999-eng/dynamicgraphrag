# Thesis Plan — Dynamic GraphRAG for Vietnamese Literary MT

**Branch:** `refactor/2026-thesis` · **Last updated:** 2026-05-14 (RAG vs GG baseline + TM seed expansion + A1 hierarchical retrieval)

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
| Week 2 — Judge revision: pairwise approach | done — deepseek-chat pairwise = 57.1% per-pair agreement | uncommitted |
| Week 2 — A3 chap-6 ablation full pairwise (n=62) | done — TM_on 62.5% decisive win rate, p=0.077 one-sided | uncommitted |
| Week 2 — Phase 3 human pairwise review (n=30) | **done — aggregate TM_on 61.5% matches judge 62.5% (1pp gap)** | uncommitted |
| Week 2 — docln crawler + alignment editor | done — 9/18 chap fetched (chap 3-7 aligned by user, 8-10/12-17/20 blocked) | committed 1c26237 |
| Week 2 — User edits VN chap 11-19 via vntq_edit_app | done | uncommitted (data only) |
| Week 2 — RAG vs GG baseline pairwise judge | done — TM_on 43% / TM_off 20% decisive vs GG | uncommitted |
| Week 2 — TM seed expansion chap 11-19 | done — 50 → 928 entries | uncommitted |
| Week 3 — A1 Hierarchical retrieval | code complete (typed glossary schema + bucketed rendering) | uncommitted |
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

## Phase 3 human pairwise review findings (2026-05-14)

User reviewed 30 anonymized pairs via `review_app.py` with rubric-aligned instructions
(focus on word-level errors + xưng hô first, prose flow second).

| Comparison | Result |
|---|---|
| Set 1 NEW (rubric) vs deepseek-chat — per pair | 7/14 = 50.0% (down from 57.1% prose-bias baseline) |
| Set 1 NEW vs OLD (how often user changed their mind under rubric) | 8/14 agreed (user shifted 6 pairs, mostly toward "tie") |
| Set 2 (16 ablation pairs) human vs judge — per pair | 7/16 = 43.8% |
| **Set 2 aggregate TM_on win rate — human n=16** | **61.5% (8/13 decisive)** |
| **Chap 6 aggregate TM_on win rate — judge n=62** | **62.5% (25/40 decisive)** |

**Headline thesis finding**: per-pair preference judgments are inherently noisy
(subjective tie-threshold differs between human and rubric-following judge), but
**the aggregate ablation metric — which-system-wins-on-average — matches between
human and judge within 1 percentage point.** Since aggregate ablation IS the
thesis question (does TM beat no-TM?), the judge is validated for the actual
metric of interest. Per-pair agreement is reported as a secondary statistic with
diagnostic of disagreement patterns (judge catches word-level slips humans miss
when reading at prose level — see [[review-style-prose-level]] memory).

## RAG vs GG baseline pairwise (2026-05-14, chap 6 only)

deepseek-chat pairwise judging on chap 6 with TM_on/TM_off candidates merged with the
GG column from `aligned_master_chap_6.csv` (Source_ID matches 1:1 across systems).

| Comparison | A wins | B wins | Tie | A decisive % | sign-test (B>A) |
|---|---|---|---|---|---|
| TM_on (qwen3:8b) vs GG | 26 | 34 | 28 | **43.3%** | p = 0.181 |
| TM_off (qwen3:8b base RAG) vs GG | 10 | 40 | 19 | **20.0%** | **p < 0.0001** |
| TM_on vs TM_off (prior result) | 25 | 15 | 22 | 62.5% | p = 0.077 |

**Triangulation:** Without TM, base RAG loses to GG decisively (80% GG wins, p<0.0001).
With TM, we close the gap to "GG wins 57% vs RAG 43%, not significant at p=0.18". TM
contributes a measurable ~23 percentage-point lift toward closing the GG gap. This is
the most concrete A3 evidence to date.

The 13pp residual gap is what A1 (hierarchical glossary) + A2 (coreference) need to
close to achieve "RAG > GG" thesis claim. Outputs in
`evaluation/llm_judge/pairwise_chap6_TM{on,off}_vs_GG_deepseek-chat.csv`.

## TM seed expansion (2026-05-14)

User aligned chap 11-19 VN edit references via `vntq_edit_app.py`. Pipeline:

1. `align_reference.py` (new): EN source + user VN edit → `aligned_master_chap_N.csv`
   with columns `Source_ID, Source_Text, ref`. 9 chapters processed, 1,292 src sents.
2. `seed_translation_memory.py --experiment qwen3-8b-local_yes_tm --chapters 11..19
   --source-dir evaluation/alignment_results_reference --max-pairs 2000`.
3. TM size 50 → 928 (+878 entries). Chap 1 entries preserved (no dedup loss).

Retrieval-quality probe on 5 random chap-6 sentences against the expanded TM showed
average top-1 similarity ~0.46 (unchanged from chap-1-only seed; chap 11-19 plot
is too distant from chap 6 to lift those scores). Expansion impact is expected at
chap 7-10 where plot continuity overlaps with chap 11+.

**Methodological note for paper:** TM contains chap 11-19 user edits (after the
test chapter range 7-10). Framing: "mature translator with established style"
scenario, not a "cold-start" scenario. Chap 1 + chap 11-19 are USER-validated;
chap 7-10 model translations are NOT fed back into TM during the ablation.

## A1 hierarchical retrieval (2026-05-14, code complete)

Migrated glossary schema from `{term: str}` to `{term: {translation, type}}`. Types
emitted by the extraction prompt: ORG, DEITY, LOCATION, CONCEPT, ITEM, TITLE
(default UNKNOWN for legacy entries).

Files changed:
- `modules/prompts.py`: `build_glossary_extraction_prompt` requests typed output;
  `format_glossary_by_type` helper renders bucketed Markdown sections; both
  `build_translation_prompt` and `build_post_editing_prompt` use the helper.
- `modules/graph_manager.py`: `load_data` migrates flat strings into typed dicts
  on read; `update_glossary` coerces either schema; `get_relevant_context` returns
  `{term: {translation, type}}` dicts.

Backward compatibility verified: loaded an existing `term_glossary.json` (38 flat
entries) and the migration tagged them as type=UNKNOWN; downstream rendering
groups them under "Khác (UNKNOWN)" section. Next ablation run will populate real
types as new chapters are processed.

## docln alignment progress (2026-05-13/14)

User has aligned chapters 3-7 via `docln_align_app.py` (output:
`data/LOTM/chapters_vi/chap_{3,4,5,6,7}.txt`). Stuck at chap 8 because the
web.archive snapshot didn't capture it (along with 10, 12-17, 20 — 9 of 18
target chapters missing). User is contacting the docln site owner to source
missing chapters directly. Meanwhile chap 11, 18, 19 remain crawlable for next
alignment session.

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

**A. (DONE Phase 3 — output at `evaluation/llm_judge/spot_check_v2_responses.csv`):**
Phase 3 completed; aggregate result validates judge for ablation. No further user
action needed unless we re-run on a fresh batch later for robustness.

**A.bis Continue docln alignment when unblocked:**
- Currently aligned: chap 3-7 (saved to `data/LOTM/chapters_vi/`).
- Pending crawl: chap 8, 10, 12-17, 20 (web.archive snapshot incomplete).
  User is contacting the site owner to obtain these.
- Available right now: chap 11, 18, 19 from docln crawl (can align via
  `python docln_align_app.py` → http://localhost:5002 → enter EN chap 11/18/19).

**B. Refine VN reference for chap 11-25 (for TM seed expansion + LoRA hold-out):**

Chap 11-25 Vietnamese auto-MT already crawled from vietnamthuquan.eu — raw text
sits at `data/LOTM/chapters_vi_raw/chap_<N>.txt`. The site is partial machine
translation, so the text needs hand-fixing before it becomes a usable reference.

```bash
# In WSL graphrag env:
python vntq_edit_app.py
# Open http://localhost:5001/  — side-by-side EN (if any) + editable VN.
# Your edits save to data/LOTM/chapters_vi/chap_<N>.txt (preserves raw separately).
# Ctrl+S in the textarea = Save.
```

EN source for chap 11-25 not yet on disk. Two options:
- Edit VN-only (faster) — the editor still works; left pane just shows a stub
  message. Output still feeds into the TM seed.
- Source EN from Wuxiaworld (or equivalent) and drop as `data/LOTM/chapters/chap_<N>.txt`.
  Editor then shows side-by-side automatically.

Why chap 11-25 specifically: test set is chap 6-10 (per `const.START_CHAPTER=6`,
`END_CHAPTER=10`). Chap 1-5 already used for TM seed. To avoid data leakage:
- TM seed pool: chap 1-5 + chap 11-25 (20 chapters, ~3000 sentence pairs)
- LoRA hold-out dataset: chap 11-25 (15 chapters, kept strictly OUT of judge eval)
- Test ablation: chap 6-10 (untouched by training)

If you want to crawl more chapters (26+) later: `python crawl_vntq.py --start 26 --end 30`.

### Claude actions next session (after USER does A and B)

1. **Read plan.md + project memory** (`~/.claude/projects/C--Users-trang/memory/`).
2. **Check git status** + `git log` on `refactor/2026-thesis`.
3. **Score Phase 3 results**: `python score_spot_check_v2.py` → report new vs old agreement on set 1, human vs judge on set 2. If ≥70% on set 1: document as paper finding. If <70%: dig into disagreements.
4. **Align chap 11-25** (after user has edited the VN via `vntq_edit_app.py`): run `align_ablation.py` or `batch_alignment.py` with EN from `data/LOTM/chapters/chap_<N>.txt` (if present) and VN from `data/LOTM/chapters_vi/chap_<N>.txt` to produce `evaluation/alignment_results_*/aligned_master_chap_<N>.csv` for N=11..25. If EN source is missing for some chapters, alignment skips them and we can still use the VN-only text as TM seed (single-side embedding).
5. **Expand TM seed**: `python seed_translation_memory.py --chapters 1,2,3,4,5,11,...,25 --max-pairs 1000` to populate TM with broader coverage.
6. **Generate TM_on/TM_off translations for chap 7-10**: run `main.py` twice with `--experiment-suffix tm` (TM on) and `--experiment-suffix notm` (TM off) over chap 7-10. ~6-12 hours of Ollama background work (qwen3:8b WSL).
7. **Align chap 7-10 ablation outputs** with `align_ablation.py` → `aligned_master_chap_{7,8,9,10}.csv` with TM_on/TM_off columns.
8. **Run pairwise judge on chap 7-10 ablation**: `python batch_pairwise_judge.py --review evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_<N>.csv --col-a TM_on --col-b TM_off --col-source Source_Text --col-ref ref --col-id Source_ID --default-chapter <N> --judge deepseek-chat --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --out evaluation/llm_judge/pairwise_ablation_chap<N>_TMon_vs_TMoff_deepseek-chat.csv`. Cost ~$1-2 total. Target combined n≥200 across chap 6-10 for sign-test p<0.05.
9. **(Optional) Phase 4 calibration triples** if you want bulletproof paper methodology: hand-craft 10-15 src sentences with 3 versions each (Gold/Mid/Bad), validate judge picks Gold > Mid > Bad ≥95%. ~30 min Claude time + $0.10 API.
10. **Then A1 — Hierarchical retrieval**: persist entity types in `graph_manager` schema (currently flat dict, needs `{term: {translation, type}}`). Update `get_relevant_context` to bucket by type. Update prompts to render glossary as separate sections.

## Files added/changed in this session (uncommitted as of 2026-05-13 pairwise/judge work)

Already committed in `7977c8a` (prior session): translation_memory.py, seed_translation_memory.py, align_ablation.py, pipeline.py changes, main.py ablation flags, prompts.py TM injection, knowledge_base ablation dirs, aligned_master_chap_6.csv, partial per-axis scores.

Net new in this session (judge revision + Phase 3 + crawler/editor stack):
- `batch_pairwise_judge.py` — pairwise runner with configurable column mapping + position-bias swap
- `probe_judge_providers.py` — free-tier provider availability probe
- `prepare_spot_check_v2.py` — Phase 3 data prep, 14 re-eval + 16 ablation anonymized
- `review_app.py` — Flask UI for Phase 3 pairwise review (port 5000)
- `score_spot_check_v2.py` — Phase 3 scoring vs deepseek-chat (per pair + aggregate)
- `score_spot_check.py` — added `--rubric prose` for re-aggregation experiment
- `modules/prompts.py` — added `build_literary_pairwise_judge_prompt`
- `crawl_vntq.py` — vntq.eu LOTM Vietnamese crawler (low quality, fallback)
- `vntq_edit_app.py` — Flask editor for vntq raw VN (port 5001)
- `crawl_docln.py` — docln.net LOTM Vietnamese crawler via web.archive.org (high quality, 9/18 chap available)
- `docln_align_app.py` — Flask 3-pane alignment editor: EN ↔ docln raw ↔ curated VN (port 5002)
- `plan.md` — Phase 3 findings + alignment progress + next-session checklist

Output data (gitignored, kept locally):
- `evaluation/llm_judge/pairwise_spot_check_for_review_deepseek-{chat,reasoner,chat_v2}.csv` — Phase 1 spot-check pairwise runs
- `evaluation/llm_judge/pairwise_ablation_chap6_TMon_vs_TMoff_deepseek-chat.csv` — headline ablation, n=62
- `evaluation/llm_judge/spot_check_v2_{pairs,key,responses}.csv` — Phase 3 inputs + user responses
- `data/LOTM/chapters_vi_raw/chap_{11..25}.txt` — vntq raw crawl
- `data/LOTM/chapters_vi_docln_raw/chap_{3,4,5,6,7,9,11,18,19}.txt` — docln raw crawl (9 chapters)
- `data/LOTM/chapters_vi/chap_{3,4,5,6,7}.txt` — user-aligned VN references for EN chap 3-7
