from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)

from local_llm import LLM, GenConfig

from jd.schemas import (
    JobDescriptionAnalysis,
)

from matching.evidence_map import (
    EvidenceMap,
)


ResumeSection = Literal[
    "summary",
    "experience",
    "projects",
    "education",
    "certifications",
]


class DraftPlannedBullet(BaseModel):

    section: ResumeSection

    objective: str

    requirement_ids: list[
        str
    ] = Field(default_factory=list)

    evidence_ids: list[
        str
    ] = Field(default_factory=list)


class PlannerResponse(BaseModel):

    summary_focus: list[
        str
    ] = Field(default_factory=list)

    skills_to_emphasize: list[
        str
    ] = Field(default_factory=list)

    bullets: list[
        DraftPlannedBullet
    ] = Field(default_factory=list)


class PlannedBullet(
    DraftPlannedBullet
):

    plan_id: str


class ResumePlan(BaseModel):

    target_role: str | None

    summary_focus: list[str]

    skills_to_emphasize: list[str]

    bullets: list[
        PlannedBullet
    ]


SYSTEM_PROMPT = """
You are the planning stage of an evidence-grounded resume generator.

Plan the strongest truthful resume for the target role.

RULES:

1. Use only evidence IDs supplied to you.
2. Never plan a claim unsupported by evidence.
3. Prioritize required job requirements.
4. Do not attempt to hide missing requirements with invented experience.
5. Every planned bullet must reference at least one evidence ID.
6. Prefer measurable, technically specific evidence.
7. Avoid redundant bullets.
8. Return JSON only.

Format:

{
  "summary_focus": ["Python backend development"],
  "skills_to_emphasize": ["Python", "SQL"],
  "bullets": [
    {
      "section": "projects",
      "objective": "Demonstrate Python API development",
      "requirement_ids": ["req_x"],
      "evidence_ids": ["ev_x"]
    }
  ]
}
"""


def make_plan_id(
    bullet: DraftPlannedBullet,
) -> str:

    raw = (
        bullet.section
        + bullet.objective
        + "|".join(
            bullet.evidence_ids
        )
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return f"plan_{digest[:12]}"


def plan_resume(
    llm: LLM,
    analysis: JobDescriptionAnalysis,
    evidence_map: EvidenceMap,
    evidence_lookup: dict[
        str,
        dict,
    ],
) -> ResumePlan:

    usable_evidence_ids = set()

    requirement_payload = []

    for mapped in (
        evidence_map.requirements
    ):

        usable_evidence_ids.update(
            mapped.evidence_ids
        )

        requirement_payload.append(
            mapped.model_dump()
        )

    evidence_payload = []

    supported_skills = set()

    for evidence_id in (
        usable_evidence_ids
    ):

        card = evidence_lookup.get(
            evidence_id
        )

        if not card:
            continue

        supported_skills.update(
            card.get(
                "skills",
                [],
            )
        )

        evidence_payload.append(
            {
                "evidence_id":
                    evidence_id,

                "claim":
                    card.get("claim"),

                "category":
                    card.get("category"),

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

    payload = {
        "target_role":
            analysis.role_title,

        "requirements":
            requirement_payload,

        "evidence":
            evidence_payload,
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
            max_new_tokens=1600,
            temperature=0.0,
        ),
    )

    start = raw.find("{")
    end = raw.rfind("}")

    parsed = json.loads(
        raw[start:end + 1]
    )

    response = (
        PlannerResponse
        .model_validate(parsed)
    )

    bullets = []

    for bullet in response.bullets:

        valid_evidence = [
            evidence_id
            for evidence_id
            in bullet.evidence_ids
            if evidence_id
            in usable_evidence_ids
        ]

        if not valid_evidence:
            continue

        bullet.evidence_ids = (
            valid_evidence
        )

        bullets.append(
            PlannedBullet(
                **bullet.model_dump(),
                plan_id=(
                    make_plan_id(
                        bullet
                    )
                ),
            )
        )

    valid_skills = [
        skill
        for skill
        in response.skills_to_emphasize
        if skill.lower()
        in {
            supported.lower()
            for supported
            in supported_skills
        }
    ]

    return ResumePlan(
        target_role=(
            analysis.role_title
        ),
        summary_focus=(
            response.summary_focus
        ),
        skills_to_emphasize=(
            valid_skills
        ),
        bullets=bullets,
    )