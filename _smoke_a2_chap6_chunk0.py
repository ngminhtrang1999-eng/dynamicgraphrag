"""A2 wiring smoke test — no Ollama, no LLM call.

Runs on chap 6 chunk 0:
  1. Load CoreferenceResolver
  2. Inject entity tags with the deterministic A2 path
  3. Show: clusters found, canonical names, injected_ref text, entities_mentioned
  4. Compare side-by-side with the legacy baseline (add_placeholder_regex + spaCy candidates)

Pass criteria (eyeball):
  - >=1 cluster found
  - At least one cluster has canonical 'Klein' (or similar proper noun)
  - injected_ref contains [Klein]-style tags after pronouns
  - entities_mentioned is non-empty
"""
import re
import sys

LOG_PATH = "_smoke_a2_chap6_chunk0.log"


def _redirect_to_log():
    """Tee stdout+stderr to LOG_PATH. Terminal stays clean; file captures
    everything including fastcoref's model-load progress so Claude can read
    it back without the user copy-pasting."""
    log = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log


_redirect_to_log()

# Imports AFTER redirect so any model-load chatter from fastcoref also lands
# in the log.
from utils import split_text_with_overlap, get_spacy_candidates, filter_similar_names, add_placeholder_regex
import const
from modules.coreference import CoreferenceResolver


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"

    src_path = "data/LOTM/chapters/chapter_0006.txt"
    print(f"📖 Loading {src_path}...")
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\s+", " ", text)

    chunks = split_text_with_overlap(text, const.CHUNK_SIZE, 0)
    chunk = chunks[0]
    print(f"📄 Chunk 0 length: {len(chunk)} chars, {len(chunks)} total chunks in chap 6.")
    print()
    print("=" * 70)
    print("CHUNK 0 SOURCE (first 400 chars):")
    print("=" * 70)
    print(chunk[:400], "...")
    print()

    # === Baseline path (spaCy + add_placeholder_regex) ===
    print("=" * 70)
    print("BASELINE (spaCy NER + add_placeholder_regex):")
    print("=" * 70)
    spacy_candidates = filter_similar_names(get_spacy_candidates(chunk))
    print(f"spaCy candidates: {spacy_candidates}")
    baseline_with_placeholders = add_placeholder_regex(chunk)
    num_brackets = baseline_with_placeholders.count("[]")
    print(f"Empty []  placeholders: {num_brackets}")
    print()
    print(f"Baseline first 400 chars (would then be filled by LLM injection call):")
    print(baseline_with_placeholders[:400], "...")
    print()

    # === A2 path ===
    print("=" * 70)
    print(f"A2 (fastcoref on device={device}):")
    print("=" * 70)
    resolver = CoreferenceResolver(device=device)
    print("Loading fastcoref model...")
    resolver._ensure_loaded()
    print(f"Loaded on device={resolver.device}.")
    print()

    result = resolver.resolve(chunk)
    print(f"Clusters: {len(result.clusters)}")
    for i, (cluster, name) in enumerate(zip(result.clusters, result.canonical)):
        mentions = [chunk[s:e] for s, e in cluster]
        # truncate display
        display_mentions = [m if len(m) < 30 else m[:27] + "..." for m in mentions]
        print(f"  [{i}] canonical={name!r:>25}  mentions={display_mentions}")
    print()
    print(f"entities_mentioned() -> {result.entities_mentioned()}")
    print()

    injected_ref, _ = resolver.inject_entity_tags(chunk)
    tag_count = injected_ref.count("[")
    empty_tag_count = injected_ref.count("[]")
    named_tag_count = tag_count - empty_tag_count
    print(f"injected_ref: total tags={tag_count}, named={named_tag_count}, empty=[]={empty_tag_count}")
    print()
    print("Injected_ref first 600 chars:")
    print(injected_ref[:600], "...")
    print()

    print("=" * 70)
    print("PASS CRITERIA CHECK:")
    print("=" * 70)
    ok_clusters = len(result.clusters) >= 1
    ok_canonical = any(result.canonical)
    ok_entities = bool(result.entities_mentioned())
    ok_tags = named_tag_count >= 1
    klein_seen = any("Klein" in c or "Zhou" in c for c in result.canonical)
    for label, ok in [
        (">= 1 cluster", ok_clusters),
        ("at least one canonical name", ok_canonical),
        ("entities_mentioned() non-empty", ok_entities),
        (">= 1 named [Name] tag emitted", ok_tags),
        ("'Klein' or 'Zhou' surfaced as canonical (chap 6 main char)", klein_seen),
    ]:
        print(f"  {'✅' if ok else '❌'} {label}")
    overall = all([ok_clusters, ok_canonical, ok_entities, ok_tags])
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
