from __future__ import annotations

from retrieval.bm25 import (
    BM25Retriever,
)

from retrieval.common import (
    SearchResult,
)

from retrieval.semantic import (
    SemanticRetriever,
)


class HybridRetriever:
    """
    Combine sparse and dense retrieval
    using weighted Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        semantic: SemanticRetriever,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:

        self.bm25 = bm25
        self.semantic = semantic

        self.bm25_weight = (
            bm25_weight
        )

        self.semantic_weight = (
            semantic_weight
        )

        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 15,
        candidate_k: int = 30,
    ) -> list[SearchResult]:

        bm25_results = (
            self.bm25.search(
                query,
                top_k=candidate_k,
            )
        )

        semantic_results = (
            self.semantic.search(
                query,
                top_k=candidate_k,
            )
        )

        fused: dict[
            str,
            SearchResult,
        ] = {}

        def get_id(
            result: SearchResult,
        ) -> str:

            return str(
                result.card[
                    "evidence_id"
                ]
            )

        # -------------------------------------------------------------
        # BM25
        # -------------------------------------------------------------

        for result in bm25_results:

            evidence_id = get_id(
                result
            )

            current = fused.setdefault(
                evidence_id,
                SearchResult(
                    card=result.card
                ),
            )

            current.bm25_score = (
                result.bm25_score
            )

            current.bm25_rank = (
                result.bm25_rank
            )

            contribution = (
                self.bm25_weight
                / (
                    self.rrf_k
                    + result.bm25_rank
                )
            )

            current.fusion_score = (
                current.fusion_score
                or 0.0
            ) + contribution

        # -------------------------------------------------------------
        # Semantic
        # -------------------------------------------------------------

        for result in semantic_results:

            evidence_id = get_id(
                result
            )

            current = fused.setdefault(
                evidence_id,
                SearchResult(
                    card=result.card
                ),
            )

            current.semantic_score = (
                result.semantic_score
            )

            current.semantic_rank = (
                result.semantic_rank
            )

            contribution = (
                self.semantic_weight
                / (
                    self.rrf_k
                    + result.semantic_rank
                )
            )

            current.fusion_score = (
                current.fusion_score
                or 0.0
            ) + contribution

        ranked = sorted(
            fused.values(),

            key=lambda result: (
                result.fusion_score
                or 0.0
            ),

            reverse=True,
        )

        return ranked[:top_k]