from __future__ import annotations

import math
import re
from collections import Counter

from retrieval.common import (
    SearchResult,
    card_to_text,
)


TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*"
)


def tokenize(
    text: str,
) -> list[str]:

    return [
        token.lower()
        for token
        in TOKEN_PATTERN.findall(text)
    ]


class BM25Retriever:
    """
    Small in-memory Okapi BM25 retriever.

    ResumeForge's evidence corpus will be relatively
    small, so an in-memory implementation is enough.
    """

    def __init__(
        self,
        cards: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:

        self.cards = cards
        self.k1 = k1
        self.b = b

        self.documents = [
            tokenize(
                card_to_text(card)
            )
            for card in cards
        ]

        self.term_frequencies = [
            Counter(document)
            for document
            in self.documents
        ]

        self.document_lengths = [
            len(document)
            for document
            in self.documents
        ]

        self.document_count = len(
            self.documents
        )

        self.average_document_length = (
            sum(self.document_lengths)
            / self.document_count

            if self.document_count

            else 0.0
        )

        self.document_frequencies = (
            self._build_document_frequencies()
        )

    def _build_document_frequencies(
        self,
    ) -> Counter[str]:

        frequencies = Counter()

        for document in self.documents:
            frequencies.update(
                set(document)
            )

        return frequencies

    def _idf(
        self,
        term: str,
    ) -> float:

        document_frequency = (
            self.document_frequencies.get(
                term,
                0,
            )
        )

        return math.log(
            1.0
            + (
                self.document_count
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

    def _score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:

        if (
            self.average_document_length
            == 0
        ):
            return 0.0

        frequencies = (
            self.term_frequencies[
                document_index
            ]
        )

        document_length = (
            self.document_lengths[
                document_index
            ]
        )

        score = 0.0

        for term in set(
            query_tokens
        ):

            term_frequency = (
                frequencies.get(
                    term,
                    0,
                )
            )

            if term_frequency == 0:
                continue

            numerator = (
                term_frequency
                * (self.k1 + 1.0)
            )

            denominator = (
                term_frequency
                + self.k1
                * (
                    1.0
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self.average_document_length
                    )
                )
            )

            score += (
                self._idf(term)
                * numerator
                / denominator
            )

        return score

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[SearchResult]:

        if not self.cards:
            return []

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        scored = [
            (
                index,
                self._score_document(
                    query_tokens,
                    index,
                ),
            )
            for index in range(
                len(self.cards)
            )
        ]

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for rank, (
            index,
            score,
        ) in enumerate(
            scored[:top_k],
            start=1,
        ):

            if score <= 0:
                continue

            results.append(
                SearchResult(
                    card=self.cards[index],
                    bm25_score=float(
                        score
                    ),
                    bm25_rank=rank,
                )
            )

        return results