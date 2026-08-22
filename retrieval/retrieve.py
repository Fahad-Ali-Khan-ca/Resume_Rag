from __future__ import annotations

import argparse

from retrieval.bm25 import (
    BM25Retriever,
)

from retrieval.common import (
    load_evidence,
)

from retrieval.hybrid import (
    HybridRetriever,
)

from retrieval.reranker import (
    EvidenceReranker,
)

from retrieval.semantic import (
    SemanticRetriever,
)


def build_retrieval_pipeline(
    device: str | None = None,
) -> tuple[
    HybridRetriever,
    EvidenceReranker,
]:

    cards = load_evidence()

    bm25 = BM25Retriever(
        cards
    )

    semantic = SemanticRetriever(
        cards,
        device=device,
    )

    hybrid = HybridRetriever(
        bm25=bm25,
        semantic=semantic,
    )

    reranker = EvidenceReranker(
        device=device,
    )

    return hybrid, reranker


def search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    device: str | None = None,
):

    hybrid, reranker = (
        build_retrieval_pipeline(
            device=device
        )
    )

    candidates = hybrid.search(
        query=query,

        top_k=candidate_k,

        candidate_k=max(
            candidate_k,
            30,
        ),
    )

    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Search ResumeForge's "
            "evidence corpus."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Requirement or evidence query."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional device: "
            "cuda or cpu."
        ),
    )

    args = parser.parse_args()

    results = search(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        device=args.device,
    )

    print()
    print("ResumeForge Retrieval")
    print("---------------------")
    print(
        f"Query: {args.query}"
    )
    print()

    if not results:
        print(
            "No evidence found."
        )
        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        card = result.card

        print(
            f"{index}. "
            f"{card['claim']}"
        )

        print(
            f"   Evidence: "
            f"{card['evidence_id']}"
        )

        print(
            f"   Category: "
            f"{card['category']}"
        )

        if card.get("skills"):
            print(
                "   Skills: "
                + ", ".join(
                    card["skills"]
                )
            )

        print(
            "   Scores: "
            f"BM25="
            f"{result.bm25_score}, "
            f"Semantic="
            f"{result.semantic_score}, "
            f"Fusion="
            f"{result.fusion_score}, "
            f"Rerank="
            f"{result.rerank_score}"
        )

        print(
            f"   Source: "
            f"{card['source_file']}"
        )

        if (
            card.get("page")
            is not None
        ):
            print(
                f"   Page: "
                f"{card['page']}"
            )

        print()


if __name__ == "__main__":
    main()