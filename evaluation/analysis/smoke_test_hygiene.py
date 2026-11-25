"""End-to-end sanity check before kicking off the chap 6-10 hygiene run.

Verifies that the cloned + corrected experiment dir loads cleanly through
the same code paths the pipeline will use:
  - GraphManager.load_data: glossary corrections visible
  - TranslationMemory: Zhou Mingrui rewrites visible
  - build_draft_translation_prompt / build_post_editing_prompt: hard_rules present
  - output_filter.contamination_score: flags a known leak example
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import os
os.chdir(REPO_ROOT)

from modules.graph_manager import GraphManager
from modules.translation_memory import TranslationMemory
from modules.output_filter import contamination_score
from modules.prompts import build_draft_translation_prompt, build_post_editing_prompt

EXP_DIR = "data/knowledge_base/qwen3-8b-local_yes_hygiene"

# 1. GraphManager loads
gm = GraphManager(EXP_DIR)
gm.load_data()
print(f"✓ GraphManager loaded. glossary terms: {len(gm.glossary)}, "
      f"last_processed_chap={gm.tracking['last_processed_chap']}")

for term in ["Mr. Fool", "Beyonder", "gray fog", "Nighthawks", "Zhou Mingrui", "Klein"]:
    if term in gm.glossary:
        e = gm.glossary[term]
        print(f"  {term!r:30s} -> {e}")
    else:
        print(f"  {term!r:30s} -> MISSING")

# 2. TM loads + lookup
tm = TranslationMemory(EXP_DIR)
print(f"✓ TM loaded: {len(tm)} entries")

probe = "Zhou Mingrui slowly opened his eyes."
hits = tm.retrieve(probe, k=3, min_similarity=0.1)
print(f"  TM hits on probe {probe!r}: {len(hits)}")
for h in hits[:2]:
    print(f"    sim={h['similarity']:.3f}")
    print(f"      src   = {h['source'][:80]!r}")
    print(f"      trans = {h['translation'][:90]!r}")
    if "Zhou Mingrui" in h["translation"]:
        print("      ⚠️ pinyin still present — rewrite missed this entry")
    if "Chu Minh Thụy" in h["translation"]:
        print("      ✓ rewrite applied: contains 'Chu Minh Thụy'")

# 3. Prompt assertions
draft = build_draft_translation_prompt("Zhou Mingrui opened his eyes.", "", tm_examples=None)
edit = build_post_editing_prompt("src", "draft", "src", dict(list(gm.glossary.items())[:3]), [])
assert "<hard_rules>" in draft[0]["content"], "draft missing hard_rules"
assert "<hard_rules>" in edit[0]["content"], "edit missing hard_rules"
assert "Chu Minh Thụy" in draft[0]["content"], "draft missing name policy example"
print("✓ Prompts contain hard_rules with name policy.")

# 4. Contamination filter
score, br = contamination_score("Zhou Mingrui nhìn lên trời, kinh悚 và lo lắng.")
assert score > 0, "expected contamination on test sentence"
print(f"✓ Contamination filter: score={score} ({br})")

print("\nAll checks passed. Hygiene dir is ready for generation.")
