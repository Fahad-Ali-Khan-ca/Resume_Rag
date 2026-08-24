from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jd.schemas import (
    JobDescriptionAnalysis,
    JobRequirement,
)

from retrieval.hybrid import (
    HybridRetriever,
)

from retrieval.reranker import (
    EvidenceReranker,
)


class CandidateEvidence(BaseModel):

    card: dict[str, Any]

    bm25_score: float | None = None
    semantic_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class RequirementCandidates(BaseModel):

    requirement_id: str
    requirement_text: str

    candidates: list[
        CandidateEvidence
    ] = Field(default_factory=list)


class EvidenceMatcher:

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: EvidenceReranker,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker

    def match(
        self,
        requirement: JobRequirement,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> RequirementCandidates:

        candidates = (
            self.retriever.search(
                query=(
                    requirement.search_query
                ),
                top_k=candidate_k,
                candidate_k=max(
                    candidate_k,
                    30,
                ),
            )
        )

        candidates = (
            self.reranker.rerank(
                query=(
                    requirement.search_query
                ),
                candidates=candidates,
                top_k=top_k,
            )
        )

        converted = []

        for result in candidates:

            converted.append(
                CandidateEvidence(
                    card=result.card,
                    bm25_score=(
                        result.bm25_score
                    ),
                    semantic_score=(
                        result.semantic_score
                    ),
                    fusion_score=(
                        result.fusion_score
                    ),
                    rerank_score=(
                        result.rerank_score
                    ),
                )
            )

        return RequirementCandidates(
            requirement_id=(
                requirement.requirement_id
            ),
            requirement_text=(
                requirement.text
            ),
            candidates=converted,
        )

    def match_all(
        self,
        analysis: JobDescriptionAnalysis,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[
        RequirementCandidates
    ]:

        return [
            self.match(
                requirement,
                top_k=top_k,
                candidate_k=candidate_k,
            )
            for requirement
            in analysis.requirements
        ]