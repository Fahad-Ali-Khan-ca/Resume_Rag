from __future__ import annotations

import json

from pydantic import (
    BaseModel,
    Field,
)

from local_llm import LLM, GenConfig

from generation.bullet_generator import (
    GeneratedBullet,
)


class BulletValidation(BaseModel):

    supported: bool

    supported_claims: list[
        str
    ] = Field(default_factory=list)

    unsupported_claims: list[
        str
    ] = Field(default_factory=list)

    reason: str


SYSTEM_PROMPT = """
You are a strict factual validator for ResumeForge.

Compare the generated resume statement against ONLY the provided evidence.

RULES:

1. Every factual assertion must be supported by evidence.
2. Metrics must match exactly.
3. Technologies must be explicitly supported.
4. Do not use outside knowledge.
5. Do not assume likely responsibilities.
6. If any meaningful claim is unsupported, mark supported=false.
7. Return JSON only.

Format:

{
  "supported": true,
  "supported_claims": ["..."],
  "unsupported_claims": [],
  "reason": "All factual claims are supported."
}
"""


def validate_bullet(
    llm: LLM,
    bullet: GeneratedBullet,
    evidence_lookup: dict[
        str,
        dict,
    ],
) -> BulletValidation:

    evidence = []

    for evidence_id in (
        bullet.evidence_ids
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

        return BulletValidation(
            supported=False,
            unsupported_claims=[
                bullet.text
            ],
            reason=(
                "No supporting evidence "
                "was available."
            ),
        )

    payload = {
        "generated_statement":
            bullet.text,

        "evidence":
            evidence,
    }

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
            max_new_tokens=500,
            temperature=0.0,
        ),
    )

    start = raw.find("{")
    end = raw.rfind("}")

    return (
        BulletValidation
        .model_validate(
            json.loads(
                raw[start:end + 1]
            )
        )
    )