from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)

from retrieval.common import (
    SearchResult,
    card_to_text,
)


INDEX_DIR = Path(
    "corpus/index"
)

EMBEDDINGS_FILE = (
    INDEX_DIR
    / "evidence_embeddings.npy"
)

INDEX_META_FILE = (
    INDEX_DIR
    / "semantic_index.json"
)


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


class SemanticRetriever:

    def __init__(
        self,
        cards: list[dict],
        model_name: str = (
            DEFAULT_EMBEDDING_MODEL
        ),
        device: str | None = None,
    ) -> None:

        self.cards = cards
        self.model_name = model_name

        self.model = (
            SentenceTransformer(
                model_name,
                device=device,
            )
        )

        self.texts = [
            card_to_text(card)
            for card in cards
        ]

        self.corpus_hash = (
            self._compute_corpus_hash()
        )

        self.embeddings = (
            self._load_or_build_embeddings()
        )

    def _compute_corpus_hash(
        self,
    ) -> str:

        hasher = hashlib.sha256()

        for card, text in zip(
            self.cards,
            self.texts,
        ):

            evidence_id = str(
                card.get(
                    "evidence_id",
                    "",
                )
            )

            hasher.update(
                evidence_id.encode(
                    "utf-8"
                )
            )

            hasher.update(b"\0")

            hasher.update(
                text.encode(
                    "utf-8"
                )
            )

            hasher.update(b"\0")

        return hasher.hexdigest()

    def _cache_is_valid(
        self,
    ) -> bool:

        if (
            not EMBEDDINGS_FILE.exists()
            or not INDEX_META_FILE.exists()
        ):
            return False

        try:
            metadata = json.loads(
                INDEX_META_FILE.read_text(
                    encoding="utf-8"
                )
            )

        except (
            json.JSONDecodeError,
            OSError,
        ):
            return False

        return (
            metadata.get(
                "model_name"
            )
            == self.model_name

            and metadata.get(
                "corpus_hash"
            )
            == self.corpus_hash

            and metadata.get(
                "count"
            )
            == len(self.cards)
        )

    def _load_or_build_embeddings(
        self,
    ) -> np.ndarray:

        INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self._cache_is_valid():

            embeddings = np.load(
                EMBEDDINGS_FILE
            )

            if (
                embeddings.shape[0]
                == len(self.cards)
            ):
                return embeddings

        if not self.texts:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            self.texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(
            np.float32
        )

        np.save(
            EMBEDDINGS_FILE,
            embeddings,
        )

        INDEX_META_FILE.write_text(
            json.dumps(
                {
                    "model_name":
                        self.model_name,

                    "corpus_hash":
                        self.corpus_hash,

                    "count":
                        len(self.cards),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return embeddings

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[SearchResult]:

        if (
            not self.cards
            or self.embeddings.size == 0
            or not query.strip()
        ):
            return []

        query_embedding = (
            self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
        ).astype(
            np.float32
        )

        # Since both sides are normalized,
        # dot product == cosine similarity.

        scores = (
            self.embeddings
            @ query_embedding
        )

        top_indices = np.argsort(
            -scores
        )[:top_k]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            results.append(
                SearchResult(
                    card=self.cards[
                        int(index)
                    ],

                    semantic_score=float(
                        scores[index]
                    ),

                    semantic_rank=rank,
                )
            )

        return results