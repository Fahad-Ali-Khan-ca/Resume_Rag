from __future__ import annotations

import json
from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
)

from llm import LLM

from jd.schemas import (
    JobDescriptionAnalysis,
)

from matching.evidence_map import (
    EvidenceMap,
)

from generation.planner import (
    plan_resume,
)

from generation.bullet_generator import (
    generate_bullet,
)

from validation.claim_validator import (
    validate_bullet,
)


class ResumeBulletRecord(BaseModel):

    section: str
    text: str

    evidence_ids: list[str]

    requirement_ids: list[str]

    source_labels: list[str]

    validated: bool = True


class TailoredResume(BaseModel):

    target_role: str | None = None
    company: str | None = None

    summary: list[
        ResumeBulletRecord
    ] = Field(default_factory=list)

    skills: list[str] = Field(
        default_factory=list
    )

    sections: dict[
        str,
        list[ResumeBulletRecord],
    ] = Field(default_factory=dict)


def source_label(
    card: dict,
) -> str:

    entity = card.get(
        "entity"
    )

    if entity:
        return str(entity)

    source_file = card.get(
        "source_file",
        "Evidence",
    )

    stem = Path(
        source_file
    ).stem

    return stem.replace(
        "_",
        " ",
    ).replace(
        "-",
        " ",
    ).title()


def generate_resume(
    generator_llm: LLM,
    analysis: JobDescriptionAnalysis,
    evidence_map: EvidenceMap,
    evidence_lookup: dict[
        str,
        dict,
    ],
    validator_llm: LLM | None = None,
    max_retries: int = 2,
) -> TailoredResume:

    validator_llm = (
        validator_llm
        or generator_llm
    )

    plan = plan_resume(
        generator_llm,
        analysis,
        evidence_map,
        evidence_lookup,
    )

    summary = []

    sections: dict[
        str,
        list[ResumeBulletRecord],
    ] = {}

    for planned in plan.bullets:

        feedback = None
        accepted = None

        for _ in range(
            max_retries + 1
        ):

            bullet = generate_bullet(
                generator_llm,
                planned,
                evidence_lookup,
                feedback=feedback,
            )

            validation = (
                validate_bullet(
                    validator_llm,
                    bullet,
                    evidence_lookup,
                )
            )

            if validation.supported:

                labels = []

                for evidence_id in (
                    bullet.evidence_ids
                ):

                    card = (
                        evidence_lookup.get(
                            evidence_id
                        )
                    )

                    if card:
                        label = source_label(
                            card
                        )

                        if label not in labels:
                            labels.append(
                                label
                            )

                accepted = (
                    ResumeBulletRecord(
                        section=(
                            bullet.section
                        ),
                        text=bullet.text,
                        evidence_ids=(
                            bullet.evidence_ids
                        ),
                        requirement_ids=(
                            bullet
                            .requirement_ids
                        ),
                        source_labels=(
                            labels
                        ),
                        validated=True,
                    )
                )

                break

            feedback = (
                validation.reason
                + "\nUnsupported claims: "
                + "; ".join(
                    validation
                    .unsupported_claims
                )
            )

        if not accepted:
            continue

        if (
            accepted.section
            == "summary"
        ):
            summary.append(
                accepted
            )

        else:
            sections.setdefault(
                accepted.section,
                [],
            ).append(
                accepted
            )

    return TailoredResume(
        target_role=(
            analysis.role_title
        ),
        company=analysis.company,
        summary=summary,
        skills=(
            plan.skills_to_emphasize
        ),
        sections=sections,
    )


def save_resume(
    resume: TailoredResume,
    output_file: Path,
) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        resume.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )