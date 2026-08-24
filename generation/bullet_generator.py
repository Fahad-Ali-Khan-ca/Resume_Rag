from __future__ import annotations

import json

from pydantic import BaseModel

from local_llm import LLM, GenConfig

from generation.planner import (
    PlannedBullet,
    ResumeSection,
)


class BulletResponse(BaseModel):
    text: str


class GeneratedBullet(BaseModel):

    section: ResumeSection

    text: str

    evidence_ids: list[str]

    requirement_ids: list[str]


SYSTEM_PROMPT = """
You write evidence-grounded resume statements.

RULES:

1. Use ONLY the supplied evidence.
2. Never invent numbers, technologies, responsibilities, scale, impact, or outcomes.
3. Preserve factual metrics exactly.
4. Use strong action-oriented resume language without exaggeration.
5. Produce one concise sentence.
6. Do not use first-person pronouns.
7. Do not add information simply because it sounds impressive.
8. Return JSON only.

Format:

{
  "text": "Built a Python REST API..."
}
"""


def normalize_text(
    text: str,
) -> str:

    text = text.strip()

    for prefix in (
        "- ",
        "• ",
        "* ",
    ):
        if text.startswith(prefix):
            text = text[
                len(prefix):
            ]

    return text.strip()


def generate_bullet(
    llm: LLM,
    plan: PlannedBullet,
    evidence_lookup: dict[
        str,
        dict,
    ],
    feedback: str | None = None,
) -> GeneratedBullet:

    evidence = []

    for evidence_id in (
        plan.evidence_ids
    ):

        card = evidence_lookup.get(
            evidence_id
        )

        if card:
            evidence.append(
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

    if not evidence:
        raise ValueError(
            "No valid evidence available "
            "for this bullet."
        )

    payload = {
        "section":
            plan.section,

        "objective":
            plan.objective,

        "evidence":
            evidence,
    }

    if feedback:
        payload[
            "previous_validation_feedback"
        ] = feedback

    raw = llm.generate(
        [
            {
                "role": "system",
                "content":
                    SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
                    json.dumps(
                        payload,
                        indent=2,
                    ),
            },
        ],
        config=GenConfig(
            max_new_tokens=300,
            temperature=0.0,
        ),
    )

    start = raw.find("{")
    end = raw.rfind("}")

    response = (
        BulletResponse
        .model_validate(
            json.loads(
                raw[start:end + 1]
            )
        )
    )

    return GeneratedBullet(
        section=plan.section,
        text=normalize_text(
            response.text
        ),
        evidence_ids=(
            plan.evidence_ids
        ),
        requirement_ids=(
            plan.requirement_ids
        ),
    )