"""A2 — coreference-aware entity injection + glossary filtering.

Replaces the per-chunk "spaCy NER candidates + LLM injection call" path with a
deterministic resolver:

  1. Run fastcoref (LingMess-RoBERTa) on the chunk text. Get character-span
     clusters: each cluster groups every mention that refers to the same entity.
  2. Pick a canonical proper-noun name per cluster (the longest non-pronoun
     mention; fall back to the longest mention).
  3. For each pronoun matching he|she|him|her|his, find its cluster and emit
     `[CanonicalName]` right after it — same surface form the existing
     entity-injection LLM produces, so `build_post_editing_prompt` downstream
     does not need to change.
  4. Expose the set of canonical entity names actually mentioned in the chunk
     for downstream glossary filtering, killing the substring-match false
     positives (e.g. "ear" in "bear") that the current
     GraphManager.get_relevant_context relies on.

Cost: one fastcoref forward pass per chunk (~100ms on GPU, ~500ms CPU). No
retry loop, no token budget, no LLM injection mismatch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


_PRONOUN_RE = re.compile(r"(?i)\b(he|she|him|her|his)\b")

# Capitalized words that look like names by surface form but are
# titles/honorifics or sentence-position artifacts. Excluded when scoring
# whether a mention contains a proper-noun token.
_PROPER_NOUN_STOPLIST = {
    "sir", "lord", "lady", "mister", "miss", "madam", "mr", "mrs", "ms", "dr",
    "captain", "detective", "officer", "father", "mother", "uncle", "aunt",
    "brother", "sister", "son", "daughter", "boy", "girl", "man", "woman",
    "people", "person", "guy", "lad", "child", "kid",
    "the", "a", "an", "this", "that", "these", "those",
    "his", "her", "their", "our", "my", "your", "its",
}


def _has_proper_noun_token(mention: str) -> bool:
    """True if `mention` contains at least one token that looks like a proper
    noun: TitleCase, alphabetic (allowing hyphens), length>=3, not in the
    stoplist of honorifics/articles/common nouns."""
    for raw in mention.split():
        # Trim surrounding punctuation
        tok = re.sub(r"^[^\w]+|[^\w]+$", "", raw)
        if len(tok) < 3:
            continue
        if not tok[0].isupper():
            continue
        if not tok.replace("-", "").replace("'", "").isalpha():
            continue
        if tok.lower() in _PROPER_NOUN_STOPLIST:
            continue
        return True
    return False

# Mentions that are themselves pronouns can't be canonical; skip when picking
# the head of a cluster. Kept broad to handle relative/possessive forms too.
_PRONOUN_MENTIONS = {
    "he", "she", "him", "her", "his", "hers", "himself", "herself",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself",
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "this", "that", "these", "those", "who", "whom", "which",
    "the man", "the woman", "the boy", "the girl",
}


@dataclass
class CorefResult:
    """Output of CoreferenceResolver.resolve(text)."""
    text: str
    clusters: list[list[tuple[int, int]]]   # cluster_idx -> list of (start, end) char spans
    canonical: list[str]                     # cluster_idx -> canonical name (may be "")
    char_to_cluster: dict[int, int] = field(default_factory=dict)  # char start -> cluster_idx

    def entities_mentioned(self) -> set[str]:
        """Set of canonical names whose cluster has at least one proper-noun mention."""
        return {name for name in self.canonical if name}


class CoreferenceResolver:
    """Lazy-loaded fastcoref wrapper. Construct once, reuse across chunks."""

    def __init__(self, device: str = "cuda", model_arch: Optional[str] = None):
        self.device = device
        self.model_arch = model_arch
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from fastcoref import FCoref  # heavy import, defer
        kwargs = {"device": self.device}
        if self.model_arch:
            kwargs["model_name_or_path"] = self.model_arch
        try:
            self._model = FCoref(**kwargs)
        except Exception as exc:
            if self.device != "cpu":
                print(f"   ⚠️ fastcoref failed on {self.device} ({exc}); retrying on cpu.")
                kwargs["device"] = "cpu"
                self._model = FCoref(**kwargs)
                self.device = "cpu"
            else:
                raise

    def resolve(self, text: str) -> CorefResult:
        """Run coref on `text`. Returns clusters + canonical-name table."""
        self._ensure_loaded()
        if not text or not text.strip():
            return CorefResult(text=text, clusters=[], canonical=[])

        preds = self._model.predict(texts=[text])
        pred = preds[0]
        # fastcoref returns char-offset spans when as_strings=False
        clusters: list[list[tuple[int, int]]] = pred.get_clusters(as_strings=False)

        canonical: list[str] = []
        char_to_cluster: dict[int, int] = {}
        for idx, cluster in enumerate(clusters):
            canonical.append(self._pick_canonical(text, cluster))
            for start, _end in cluster:
                char_to_cluster[start] = idx
        return CorefResult(text=text, clusters=clusters, canonical=canonical,
                           char_to_cluster=char_to_cluster)

    @staticmethod
    def _pick_canonical(text: str, cluster: list[tuple[int, int]]) -> str:
        """Pick canonical proper-noun name for a cluster.

        Returns the LONGEST mention containing a proper-noun token (TitleCase,
        non-honorific, len>=3). Empty string if the cluster has no such mention
        (anonymous cluster — downstream emits `[]` and skips glossary filter).
        """
        proper_best = ""
        for start, end in cluster:
            mention = text[start:end].strip()
            if not mention or mention.lower() in _PRONOUN_MENTIONS:
                continue
            stripped = re.sub(r"^(the|a|an|that|this|those|these)\s+", "",
                              mention, flags=re.I).strip()
            candidate = stripped or mention
            if _has_proper_noun_token(candidate) and len(candidate) > len(proper_best):
                proper_best = candidate
        return proper_best

    def inject_entity_tags(self, text: str,
                           candidate_filter: Optional[Iterable[str]] = None
                           ) -> tuple[str, CorefResult]:
        """Produce text with `[CanonicalName]` after each he|she|him|her|his
        whose cluster has a canonical name.

        Pronouns whose cluster is anonymous (no proper-noun mention) keep empty
        `[]` so downstream prompts still get the placeholder count they expect.

        If `candidate_filter` is given, only canonical names that pass the
        filter are emitted (others get `[]`); useful for restricting to graph
        glossary entries.
        """
        result = self.resolve(text)
        if not result.clusters:
            # No coref output — return text with empty placeholders so the
            # surface shape matches the LLM-injection path.
            return _PRONOUN_RE.sub(r"\g<0> []", text), result

        allowed = None
        if candidate_filter is not None:
            allowed = {c.lower() for c in candidate_filter}

        # Walk pronoun matches in order; for each, find the smallest cluster
        # span that contains the pronoun, then emit `[canonical]`.
        out: list[str] = []
        cursor = 0
        for m in _PRONOUN_RE.finditer(text):
            out.append(text[cursor:m.end()])
            cluster_idx = self._cluster_for_span(result, m.start(), m.end())
            if cluster_idx is None:
                out.append(" []")
            else:
                name = result.canonical[cluster_idx]
                if not name or (allowed is not None and name.lower() not in allowed):
                    out.append(" []")
                else:
                    out.append(f" [{name}]")
            cursor = m.end()
        out.append(text[cursor:])
        return "".join(out), result

    @staticmethod
    def _cluster_for_span(result: CorefResult, start: int, end: int) -> Optional[int]:
        """Find cluster_idx whose mention spans contain (start, end). Prefers
        an exact-match start; falls back to any span containing the pronoun."""
        if start in result.char_to_cluster:
            return result.char_to_cluster[start]
        for idx, cluster in enumerate(result.clusters):
            for s, e in cluster:
                if s <= start and end <= e:
                    return idx
        return None


__all__ = ["CoreferenceResolver", "CorefResult"]
