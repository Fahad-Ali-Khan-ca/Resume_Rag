from __future__ import annotations

from sentence_transformers import (
    CrossEncoder,
)

from retrieval.common import (
    SearchResult,
    card_to_text,
)


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)


class EvidenceReranker:

    def __init__(
        self,
        model_name: str = (
            DEFAULT_RERANKER_MODEL
        ),
        device: str | None = None,
    ) -> None:

        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: list[
            SearchResult
        ],
        top_k: int = 5,
    ) -> list[SearchResult]:

        if not candidates:
            return []

        pairs = [
            (
                query,
                card_to_text(
                    result.card
                ),
            )
            for result
            in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        for result, score in zip(
            candidates,
            scores,
        ):

            result.rerank_score = (
                float(score)
            )

        candidates.sort(
            key=lambda result: (
                result.rerank_score

                if (
                    result.rerank_score
                    is not None
                )

                else float("-inf")
            ),
            reverse=True,
        )

        return candidates[:top_k]