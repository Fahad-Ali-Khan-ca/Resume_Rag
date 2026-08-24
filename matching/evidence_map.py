from __future__ import annotations

import json
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from llm import LLM, GenConfig

from jd.schemas import (
    JobDescriptionAnalysis,
    JobRequirement,
)

from matching.matcher import (
    RequirementCandidates,
)


MatchStrength = Literal[
    "strong",
    "partial",
    "none",
]


class EvidenceMapDecision(BaseModel):

    match_strength: MatchStrength

    selected_evidence_ids: list[
        str
    ] = Field(default_factory=list)

    reason: str


class RequirementEvidence(BaseModel):

    requirement_id: str
    requirement_text: str

    importance: str

    match_strength: MatchStrength

    evidence_ids: list[str]

    reason: str


class EvidenceMap(BaseModel):

    requirements: list[
        RequirementEvidence
    ] = Field(default_factory=list)


SYSTEM_PROMPT = """
You determine whether candidate evidence actually supports a job requirement.

RULES:

1. Use ONLY the candidate evidence provided.
2. Do not infer experience not stated in the evidence.
3. STRONG means the evidence directly supports the requirement.
4. PARTIAL means the evidence supports part of the requirement.
5. NONE means the evidence does not meaningfully support it.
6. Select only evidence IDs that genuinely support the requirement.
7. Never invent evidence IDs.
8. Return JSON only.

Format:

{
  "match_strength": "strong",
  "selected_evidence_ids": ["ev_123"],
  "reason": "The evidence directly demonstrates..."
}
"""


def parse_json(
    text: str,
) -> dict:

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start == -1:
            raise

        return json.loads(
            text[start:end + 1]
        )


def build_requirement_map(
    llm: LLM,
    requirement: JobRequirement,
    candidates: RequirementCandidates,
) -> RequirementEvidence:

    if not candidates.candidates:

        return RequirementEvidence(
            requirement_id=(
                requirement.requirement_id
            ),
            requirement_text=(
                requirement.text
            ),
            importance=(
                requirement.importance
            ),
            match_strength="none",
            evidence_ids=[],
            reason=(
                "No candidate evidence "
                "was retrieved."
            ),
        )

    evidence_payload = []

    valid_ids = set()

    for candidate in (
        candidates.candidates
    ):

        card = candidate.card

        evidence_id = card[
            "evidence_id"
        ]

        valid_ids.add(
            evidence_id
        )

        evidence_payload.append(
            {
                "evidence_id":
                    evidence_id,

                "claim":
                    card.get("claim"),

                "skills":
                    card.get(
                        "skills",
                        [],
                    ),

                "metrics":
                    card.get(
                        "metrics",
                        [],
                    ),
            }
        )

    prompt = {
        "requirement":
            requirement.text,

        "skills":
            requirement.skills,

        "candidate_evidence":
            evidence_payload,
    }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                prompt,
                indent=2,
            ),
        },
    ]

    raw = llm.generate(
        messages,
        config=GenConfig(
            max_new_tokens=600,
            temperature=0.0,
        ),
    )

    try:

        decision = (
            EvidenceMapDecision
            .model_validate(
                parse_json(raw)
            )
        )

    except (
        ValidationError,
        json.JSONDecodeError,
    ):

        return RequirementEvidence(
            requirement_id=(
                requirement.requirement_id
            ),
            requirement_text=(
                requirement.text
            ),
            importance=(
                requirement.importance
            ),
            match_strength="none",
            evidence_ids=[],
            reason=(
                "Evidence judgment "
                "could not be validated."
            ),
        )

    selected = [
        evidence_id
        for evidence_id
        in decision.selected_evidence_ids
        if evidence_id in valid_ids
    ]

    strength = (
        decision.match_strength
    )

    if (
        strength != "none"
        and not selected
    ):
        strength = "none"

    if strength == "none":
        selected = []

    return RequirementEvidence(
        requirement_id=(
            requirement.requirement_id
        ),
        requirement_text=(
            requirement.text
        ),
        importance=(
            requirement.importance
        ),
        match_strength=strength,
        evidence_ids=selected,
        reason=decision.reason,
    )


def build_evidence_map(
    llm: LLM,
    analysis: JobDescriptionAnalysis,
    candidate_sets: list[
        RequirementCandidates
    ],
) -> EvidenceMap:

    candidate_lookup = {
        candidate.requirement_id:
            candidate
        for candidate
        in candidate_sets
    }

    mapped = []

    for requirement in (
        analysis.requirements
    ):

        candidates = (
            candidate_lookup.get(
                requirement.requirement_id,
                RequirementCandidates(
                    requirement_id=(
                        requirement
                        .requirement_id
                    ),
                    requirement_text=(
                        requirement.text
                    ),
                ),
            )
        )

        mapped.append(
            build_requirement_map(
                llm,
                requirement,
                candidates,
            )
        )

    return EvidenceMap(
        requirements=mapped
    )