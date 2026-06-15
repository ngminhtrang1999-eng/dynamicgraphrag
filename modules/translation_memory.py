"""LaBSE-backed translation memory for EN->VI literary MT.

Persisted per-experiment alongside graph data:
    {data_dir}/translation_memory.npy  -- float32 [N, 768], L2-normalized
    {data_dir}/translation_memory.json -- list of {source, translation, score}

Cosine similarity = dot product because embeddings are normalized.
"""

import os
import json
import numpy as np


class TranslationMemory:
    EMBED_DIM = 768
    MODEL_NAME = "sentence-transformers/LaBSE"

    def __init__(self, data_dir, model=None):
        self.data_dir = data_dir
        self.path_emb = os.path.join(data_dir, "translation_memory.npy")
        self.path_meta = os.path.join(data_dir, "translation_memory.json")
        self.entries = []
        self.embeddings = np.zeros((0, self.EMBED_DIM), dtype=np.float32)
        self._model = model
        self.load()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"   📥 Loading LaBSE for Translation Memory...")
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def load(self):
        if os.path.exists(self.path_emb):
            self.embeddings = np.load(self.path_emb).astype(np.float32)
        if os.path.exists(self.path_meta):
            with open(self.path_meta, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        if len(self.entries) != self.embeddings.shape[0]:
            print(
                f"   ⚠️ TM size mismatch ({len(self.entries)} entries vs "
                f"{self.embeddings.shape[0]} embeddings) — resetting."
            )
            self.entries = []
            self.embeddings = np.zeros((0, self.EMBED_DIM), dtype=np.float32)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        np.save(self.path_emb, self.embeddings)
        with open(self.path_meta, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def _embed(self, texts):
        emb = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return emb.astype(np.float32)

    def add(self, source, translation, score=None):
        source = source.strip()
        translation = translation.strip()
        if not source or not translation:
            return
        for i, e in enumerate(self.entries):
            if e["source"].strip() == source:
                self.entries[i] = {
                    "source": source,
                    "translation": translation,
                    "score": score,
                }
                self.embeddings[i] = self._embed([source])[0]
                return
        emb = self._embed([source])
        self.entries.append(
            {"source": source, "translation": translation, "score": score}
        )
        self.embeddings = (
            np.vstack([self.embeddings, emb]) if self.embeddings.size else emb
        )

    def add_batch(self, items):
        """items: iterable of {source, translation, score?}. Skips duplicate sources."""
        seen = {e["source"].strip() for e in self.entries}
        new_items = []
        for it in items:
            src = (it.get("source") or "").strip()
            trans = (it.get("translation") or "").strip()
            if not src or not trans or src in seen:
                continue
            new_items.append({"source": src, "translation": trans, "score": it.get("score")})
            seen.add(src)
        if not new_items:
            return 0
        new_emb = self._embed([it["source"] for it in new_items])
        self.entries.extend(new_items)
        self.embeddings = (
            np.vstack([self.embeddings, new_emb]) if self.embeddings.size else new_emb
        )
        return len(new_items)

    def retrieve(self, source, k=3, min_score=None, min_similarity=0.0):
        """Top-k by cosine similarity.

        Args:
            source: query EN text
            k: number of examples to return
            min_score: filter on entry.score (None disables)
            min_similarity: cosine cutoff (use 0 to keep all)

        Returns:
            list[{source, translation, score, similarity}] sorted desc by similarity.
        """
        if self.embeddings.shape[0] == 0 or not source.strip():
            return []

        q = self._embed([source])
        sims = (self.embeddings @ q.T).flatten()

        idx = np.arange(len(self.entries))
        if min_score is not None:
            keep = np.array(
                [
                    i
                    for i in idx
                    if (self.entries[i].get("score") or 0) >= min_score
                ],
                dtype=int,
            )
            idx = keep
        if idx.size == 0:
            return []

        cand_sims = sims[idx]
        mask = cand_sims >= min_similarity
        idx = idx[mask]
        cand_sims = cand_sims[mask]
        if cand_sims.size == 0:
            return []

        order = np.argsort(-cand_sims)[:k]
        out = []
        for j in order:
            entry_idx = int(idx[j])
            e = self.entries[entry_idx]
            out.append(
                {
                    "source": e["source"],
                    "translation": e["translation"],
                    "score": e.get("score"),
                    "similarity": float(cand_sims[j]),
                }
            )
        return out

    def retrieve_random(self, source, k=3, min_score=None, rng=None):
        """Random-exemplar baseline: pick k entries uniformly at random.

        Mirrors retrieve()'s return schema and reports the *true* cosine
        similarity of each randomly-picked exemplar to the query, so the draft
        prompt is structurally identical to retrieval mode; only the selection
        rule (uniform random vs. top-k) differs. Isolates the marginal value of
        similarity retrieval over a static random few-shot prompt.

        Args:
            source: query EN text (used only to report similarity)
            k: number of examples to return
            min_score: filter on entry.score (None disables)
            rng: optional random.Random for reproducibility (defaults to global)
        """
        import random as _random
        if self.embeddings.shape[0] == 0 or not source.strip():
            return []

        idx = np.arange(len(self.entries))
        if min_score is not None:
            idx = np.array(
                [i for i in idx if (self.entries[i].get("score") or 0) >= min_score],
                dtype=int,
            )
        if idx.size == 0:
            return []

        q = self._embed([source])
        sims = (self.embeddings @ q.T).flatten()

        picker = rng or _random
        chosen = picker.sample(list(idx), min(k, int(idx.size)))
        out = []
        for i in chosen:
            entry_idx = int(i)
            e = self.entries[entry_idx]
            out.append(
                {
                    "source": e["source"],
                    "translation": e["translation"],
                    "score": e.get("score"),
                    "similarity": float(sims[entry_idx]),
                }
            )
        return out

    def __len__(self):
        return len(self.entries)
