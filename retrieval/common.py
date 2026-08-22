from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVIDENCE_FILE = Path("corpus/processed/evidence.jsonl")


@dataclass
class SearchResult:
    card: dict[str, Any]

    bm25_score: float | None = None
    bm25_rank: int | None = None

    semantic_score: float | None = None
    semantic_rank: int | None = None

    fusion_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.card,
            "retrieval": {
                "bm25_score": self.bm25_score,
                "bm25_rank": self.bm25_rank,
                "semantic_score": self.semantic_score,
                "semantic_rank": self.semantic_rank,
                "fusion_score": self.fusion_score,
                "rerank_score": self.rerank_score,
            },
        }


def load_evidence(
    path: Path = EVIDENCE_FILE,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Evidence corpus not found: {path}\n"
            "Run evidence_extractor.py first."
        )

    cards = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                cards.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"on line {line_number}"
                ) from error

    return cards


def card_to_text(
    card: dict[str, Any],
) -> str:
    """
    Convert an evidence card into searchable text.

    We intentionally exclude provenance such as filenames
    because those fields should not influence relevance.
    """

    parts = []

    claim = card.get("claim")

    if claim:
        parts.append(
            str(claim)
        )

    category = card.get("category")

    if category:
        parts.append(
            f"Category: {category}"
        )

    skills = card.get(
        "skills",
        [],
    )

    if skills:
        parts.append(
            "Skills: "
            + ", ".join(
                map(str, skills)
            )
        )

    metric_strings = []

    for metric in card.get(
        "metrics",
        [],
    ):
        name = metric.get(
            "name",
            "",
        )

        value = metric.get(
            "value",
            "",
        )

        context = metric.get(
            "context"
        )

        metric_text = (
            f"{name}: {value}"
        ).strip(": ")

        if context:
            metric_text += (
                f" ({context})"
            )

        if metric_text:
            metric_strings.append(
                metric_text
            )

    if metric_strings:
        parts.append(
            "Metrics: "
            + "; ".join(
                metric_strings
            )
        )

    return "\n".join(parts)