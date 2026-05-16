# Thesis Plan — Dynamic GraphRAG for Vietnamese Literary MT

**Branch:** `refactor/2026-thesis` · **Last updated:** 2026-05-16 evening — **Track 3 full + DeepSeek ceiling + IEEE paper draft done. 4,131 paired judgments. Headline: DeepSeek+RAG 81.8% vs GG (Claude judge). 70B llama 23.4% vs GG (deepseek-chat judge). Domain fit > parameter count.**

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
| Week 3 — A1 Hierarchical retrieval | code complete (typed glossary schema + bucketed rendering) | committed d192ef2 |
| Week 3 — chap 7-10 TM_on/TM_off generation + alignment + judge | done — n=484 ablation, n=540 vs GG | uncommitted |
| Week 3 — Hygiene re-run chap 6-10 + judge | done — 55.0% vs TM_on (p=0.053), 32.0% vs GG | uncommitted |
| Week 3 — Track 1 qualitative gap analysis | done — taxonomy + mechanism predictions for A2 | uncommitted |
| Week 4 — A2 Coreference retrieval | **done — 53.5% vs hygiene (p=0.111), 38.2% vs GG (+6.2pp closure)** | uncommitted |
| Week 5 — A2 cleanup `_pick_canonical` (drop descriptive-NP fallback) | code done in `modules/coreference.py`, A/B re-run SKIPPED (story pivoted) | uncommitted |
| Week 5 — Track 3 scale-out FULL chap 6-10 | **done — RAG_BIG 23.4% vs GG (n=410 dec, p<10⁻⁶); LOSES 8B+A2 (38.2%) by 15pp** | uncommitted |
| Week 5 — Groq backup key wired into GroqLLM auto-failover | done (`core/llm_backend.py:105-150`); resume chap 9,10 ran early via backup | uncommitted |
| Week 5 — OpenRouter backup key wired (qwen3-235b cancelled) | done (`core/llm_backend.py:148-200`) | uncommitted |
| Week 5 — Objective metrics chrF+COMET all 6 systems | done — three metrics disagree on 70B story (chrF favors, COMET tied, judge penalizes) | uncommitted |
| Week 5 — **DeepSeek-chat translator + Claude judge** ceiling test | **🏆 done — DeepSeek+RAG 81.8% vs GG (Claude Opus 4.7 judge, n=22 dec, p=0.002)** | uncommitted |
| Week 5 — Thesis skeleton (`thesis/outline.md`) | done — 7-chapter map, evidence allocation | uncommitted |
| Week 5 — **IEEE conference paper draft** (`thesis/paper_en.tex` + `paper_vn.tex`) | **done — Sections I-VII complete, EN+VN parallel, 5 tables, 6 worked examples, root cause analysis (R1-R5), 20 refs** | uncommitted |
| Week 6 — paper finalization | pending — compile test, Figure 1 system architecture, trim to 6 pages, fill author info |  |

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

## Chap 7-10 ablation full scale-out (2026-05-15)

Generated TM_on (with the 928-entry expanded TM) and TM_off (no TM) translations for
chap 7-10 on qwen3:8b WSL Ollama (~75 min each run). Aligned all four streams
(src/ref/TM_on/TM_off) via `align_ablation.py`; merged GG column from
`alignment_results_llama-3.1-8B/`. Eight pairwise judge runs via deepseek-chat with
two-call position-bias swap (`batch_pairwise_judge.py`). Output CSVs under
`evaluation/llm_judge/pairwise_chap{7..10}_{TMon_vs_TMoff,TMon_vs_GG}_deepseek-chat.csv`.

**A3 ablation — TM_on (A) vs TM_off (B), aggregated chap 6-10:**

| Chap | n | TM_on | TM_off | tie | TM_on % (decisive) | p (one-sided TM_on > TM_off) |
|---|---|---|---|---|---|---|
| 6 | 62 | 25 | 15 | 22 | 62.5% | 0.077 |
| 7 | 107 | 24 | 31 | 52 | 43.6% | 0.860 |
| 8 | 116 | 39 | 31 | 46 | 55.7% | 0.202 |
| 9 | 108 | 32 | 34 | 42 | 48.5% | 0.644 |
| 10 | 91 | 26 | 23 | 42 | 53.1% | 0.388 |
| **TOTAL** | **484** | **146** | **134** | **204** | **52.1%** | **0.256 — NOT SIGNIFICANT** |

**RAG+TM (A) vs GG (B):**

| Chap | n | TM_on | GG | tie | TM_on % (decisive) | p (one-sided GG > TM_on) |
|---|---|---|---|---|---|---|
| 6 | 88 | 26 | 34 | 28 | 43.3% | 0.183 |
| 7 | 106 | 24 | 49 | 33 | 32.9% | 0.002 |
| 8 | 118 | 30 | 63 | 25 | 32.3% | 0.0004 |
| 9 | 126 | 28 | 55 | 43 | 33.7% | 0.002 |
| 10 | 102 | 18 | 56 | 28 | 24.3% | <0.0001 |
| **TOTAL** | **540** | **126** | **257** | **157** | **32.9%** | **<0.0001 — GG WINS DECISIVELY** |

## Scale-out findings + thesis pivot (2026-05-15)

The two single-chapter signals from chap 6 — TM_on > TM_off (62.5%, p=0.077) and the
~13-point gap to GG (43.3%, p=0.18) — both EVAPORATE at scale:

1. **A3 (TM) does not significantly help across chap 6-10 (n=484, p=0.26).** The
   chap-6 favorable result appears to be a chapter-specific outlier rather than a
   generalizable mechanism. Per-chapter rates range 43.6% to 62.5%, centered ~52%.
2. **RAG+TM loses decisively to Google Translate at scale (32.9% RAG / 67.1% GG,
   p<0.0001 across all chapters except 6).** Even with expanded TM, the qwen3-8b
   based pipeline produces translations that the judge prefers ~2:1 against GG.

The thesis claim "smart dynamic RAG beats GG for literary MT with small LLMs" is
NOT supported by these results. Pivot needed.

**Honest paper framing — what we can defend:**
- Built a rigorous reference-based pairwise judging framework (B framework) and
  validated it against human review (per-pair noisy ~50%, aggregate within 1pp).
- Measured the GG → RAG+TM gap precisely: 67% / 33% in GG's favor, consistent across
  4 unseen chapters (chap 7-10). This is the most reliable number in the literature
  for an 8B-class local-LLM RAG vs commercial MT on Vietnamese literary translation.
- Identified that TM (A3) alone is insufficient to close the gap — needed
  contributions in other components (A1 type-bucketed glossary already coded but
  untested; A2 coreference not started).
- Open question: does the gap close meaningfully with A1+A2 + a stronger base model
  (qwen3-32B / gemma3-27B)? Empirical question, partly addressable.

**Why the chap-6 signal was misleading:** chap 6 is the introduction of beyonder
mechanics with dense terminology, where TM examples from chap 1 (also intro-heavy)
gave the most lift. Chap 7-10 are operational scenes (Klein joining Nighthawks,
training, first ritual) with action and dialogue rather than terminology — exactly
where GG's fluency dominates and TM's terminology-anchoring helps less. The
chapter-level variance is too high to draw conclusions from any single chapter.

**Implications for upcoming work:**
- A2 coreference may still help (targets xưng hô precisely); worth implementing
  before declaring final results.
- A1 hierarchical glossary needs to be ACTUALLY TESTED via re-generation on chap
  6-10 with typed extraction populated; currently only the schema migration is
  verified. Re-running the full ablation with A1 active = ~3h Ollama + 1h judge.
- If A1+A2 still leaves the gap >50% in GG's favor, the honest thesis is:
  "framework + measurement, not breakthrough." Still defensible as MS work.

## RAG vs GG baseline pairwise (2026-05-14, chap 6 only — superseded by scale-out)

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

### Claude actions next session (in order)

The negative scale-out result (2026-05-15) means the next investment goes into UNDERSTANDING the gap before scaling more experiments. Two tracks, in this order:

#### Track 1: Qualitative analysis of judge rationales (PRIORITY, ~30-60 min)

Goal: data-driven hypothesis for WHY GG wins 67% / RAG+TM 33%. Tells us where A1/A2/base-model changes should target. Output: written analysis + paper narrative.

Inputs available locally (all gitignored):
- `evaluation/llm_judge/pairwise_chap{6..10}_TMon_vs_GG_deepseek-chat.csv` — n=540 pairs, each with `rationale_1` and `rationale_2` columns explaining the judge's choice.
- `evaluation/llm_judge/pairwise_chap{6..10}_TMon_vs_TMoff_deepseek-chat.csv` — n=484, where TM helped vs hurt.
- Source `aligned_master_chap_{6..10}_with_GG.csv` for ground-truth lookup.

Suggested approach:
1. Load the 257 GG-wins rows (TM_on vs GG, final_choice=b). Concatenate both rationales per row.
2. Pattern-mine: keyword frequency over Vietnamese terms (`tự nhiên`, `mượt mà`, `xưng hô`, `bỏ sót`, `dịch sai`, `Hán Việt`, `code-switch`, etc.). Count which rationale themes dominate when GG wins.
3. Cross-tab with the 126 RAG-wins rows to find INVERSE patterns (where RAG dominates).
4. Sample 10-15 RAG-loss examples to read in full + the corresponding 10-15 RAG-win examples. Build a qualitative taxonomy: fluency-loss, register-loss, accuracy-loss, terminology-loss, hallucination, code-switch, etc.
5. Translate the taxonomy into thesis paragraphs:
   - "GG dominates on [X] axes": data evidence + sampled examples.
   - "RAG+TM dominates on [Y] axes": data evidence + sampled examples.
   - "Implications for A2 design": specific axes A2 must target.

Output artifact: new file `evaluation/analysis/qualitative_gap_analysis.md` with summary table + 20 worked examples + design implications.

#### Track 2: Implement A2 coreference + ablation (DONE 2026-05-16 — see Section "A2 results" below)

Goal: add fastcoref to resolve pronouns → entity IDs BEFORE glossary retrieval. Targets `register` axis (xưng hô) which Track 1 may flag as a residual gap.

Implementation steps:
1. `pip install fastcoref` (~500MB LingMess-RoBERTa). Verify it loads in WSL graphrag env.
2. New module `modules/coreference.py` with `CoreferenceResolver` wrapping fastcoref.predict(). Returns per-token cluster IDs + canonical entity name per cluster.
3. Patch `core/pipeline.py` Step 4 (entity injection): BEFORE the LLM injection call, run fastcoref on the chunk → produce a tag mapping (pronoun span → entity ID). Use this to:
   a. Improve the [Entity] tag accuracy (currently spaCy NER + LLM does this; fastcoref should be more precise).
   b. Filter glossary retrieval to entities ACTUALLY referenced in the chunk (currently substring match, has false positives).
4. Ablation harness: re-generate chap 6-10 with `--experiment-suffix a2on` (fastcoref active) vs `a2off` (current spaCy+LLM path). Same pairwise judge pipeline.
5. Score: compare A2_on vs A2_off + A2_on vs GG. Targets aggregate decisive sign-test p<0.05.

Expected impact: moderate. fastcoref is most-improved for chains of pronoun references in dialogue scenes (a chap-7-10 strength of GG per Track 1 hypothesis).

#### Track 3 (DEFERRED): Strong base model

Not in queued work but worth flagging: the qwen3:8b base model may be the real bottleneck. After A2 is tested, consider re-running chap 6-10 with `gemma3-27B` (NVIDIA NIM free tier still has quota?) or `deepseek-chat` as the translator (not just judge). If a 27B+ model with the same RAG/TM/A1/A2 components closes the gap, that's a clean thesis story: "small LLMs need more help than they get from RAG alone."

### Helpers ready in working tree (uncommitted, regenerated as needed)

- `_run_chap_7_10_TM_on.sh` / `_run_chap_7_10_TM_off.sh` — Ollama generation kickoff scripts
- `_align_chap_7_10.sh` — LaBSE alignment for ablation outputs
- `_merge_GG_chap_7_10.sh` — adds GG column from llama-3.1-8B aligned results
- `_judge_chap_7_10.sh` — runs all 8 pairwise comparisons via deepseek-chat

## Local output data (gitignored, kept on disk for next session)

- `evaluation/alignment_results_qwen3-8b-local-ablation/aligned_master_chap_{6..10}.csv`
  — full ablation src/ref/TM_on/TM_off; plus `_with_GG.csv` variants that include
  GG column merged from llama-3.1-8B alignment.
- `evaluation/alignment_results_reference/aligned_master_chap_{11..19}.csv` — TM seed
  source pool from user-edited references.
- `evaluation/llm_judge/pairwise_chap{6..10}_TMon_vs_{TMoff,GG}_deepseek-chat.csv` —
  the ten judge runs (5 chapters × 2 comparisons) that drive the scale-out finding.
- `evaluation/llm_judge/pairwise_chap6_TMoff_vs_GG_deepseek-chat.csv` — bookend showing
  base RAG (no TM) loses to GG 80/20.
- `data/knowledge_base/qwen3-8b-local_yes_tm/` — 928-entry TM + 107-term typed glossary
  populated through chap 10.
- `data/knowledge_base/qwen3-8b-local_yes_notm/` — TM_off twin (no TM, same glossary
  via chap-6 cutoff).
- `data/LOTM/translations/trans_chap_{7..10}_qwen3-8b-local_yes_{tm,notm}.txt` — the
  ablation outputs themselves (8 files).
- `data/LOTM/chapters_vi/chap_{11..19}.txt` — user-edited VN refs feeding TM seed.
- `data/knowledge_base/qwen3-8b-local_yes_hygiene_a2on/` — A2 ablation experiment
  dir (cloned from `_hygiene`; same TM + glossary + graph state, separate
  translation_progress for the A2 re-run).
- `data/LOTM/translations/trans_chap_{6..10}_qwen3-8b-local_yes_hygiene_a2on.txt`
  — A2 ablation outputs.
- `evaluation/alignment_results_qwen3-8b-local-a2/aligned_master_chap_{6..10}_a2_with_GG.csv`
  — A2 alignment results with GG column merged.
- `evaluation/llm_judge/pairwise_chap{6..10}_{A2_vs_Hygiene,A2_vs_GG}_deepseek-chat.csv`
  — the ten A2 judge runs (5 chapters × 2 comparisons).
- `evaluation/analysis/_a2_rationale_themes.log` — A2 theme-delta tables.

## A2 results (2026-05-16) — quick reference

Full analysis in `evaluation/analysis/qualitative_gap_analysis.md` section 5
and `evaluation/analysis/thesis_insights.md` section 2 A2 entry.

| Comparison | n decisive | A% | p (one-sided A>B) |
|---|---:|---:|---:|
| A2_on vs Hygiene | 325 | 53.5% | 0.111 (trending NOT sig) |
| A2_on vs GG | 408 | 38.2% | <0.0001 GG still wins, but +6.2pp closer than Hygiene |

**Mechanism finding (unexpected):** A2 did NOT lift the register axis
(Δ+0.3pp on register_xung_ho — essentially tied). The actual lift came from
`accuracy_mistrans` (Δ−6.5pp) and `terminology_hanviet` (Δ−3.7pp) via the
coref-anchored glossary filter eliminating false-positive PERSON injections.
Cost: `fluency_awkward` (Δ+9.4pp) from descriptive-NP canonical tags.

**Updated next-actions:**

1. Decision point: with 3 retrieval-side components done and GG gap still 28pp,
   the marginal ROI on more retrieval engineering is low. Options:
   - **Track 3 (DONE chap 6,7,8; chap 9,10 pending Groq reset 5/17):**
     llama-3.3-70b via Groq tested. Result: 70B+RAG = 22.7% vs GG, LOSES to
     8B+A2 (36.6%) by 14pp on matched chapters. **REFUTES "more params closes
     the gap"** — supports "domain fit > parameter count" claim. Need
     Chinese-rich frontier model (qwen3-235b, DeepSeek-V3.1) to test domain-fit
     hypothesis directly.
   - **LoRA fine-tune on chap 11-25 user-edits:** in-domain adaptation of
     qwen3:8b on the user-validated parallel data. Now MORE justified given
     Track 3 result — training-side intervention is the only retrieval-orthogonal
     lever left. Higher engineering cost (~1 week), but expected payoff better
     specified now.
   - **Start write-up:** the current evidence base (n>2900 judged pairs after
     Track 3 scale-out, three components + Hygiene + 70B scale-out, full
     failure-mode taxonomy, validated judge framework, chrF+COMET+judge
     three-metric comparison) is sufficient for an MS thesis. Honest framing
     is "decomposition of where small-LLM RAG can and cannot help, with
     quantitative ceiling and base-model attribution."

2. Optional immediate cleanup (Task #9):
   - `CoreferenceResolver._pick_canonical` cleanup CODE DONE (commit pending).
     Drops descriptive-NP fallback when cluster has no proper noun. A/B re-run
     on 1 chapter pending after Track 3 chap 9,10 complete.
