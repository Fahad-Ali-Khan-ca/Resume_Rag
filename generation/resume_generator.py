from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
)

from local_llm import LLM

from jd.schemas import (
    JobDescriptionAnalysis,
)

from matching.evidence_map import (
    EvidenceMap,
)

from generation.candidate_profile import (
    CandidateProfile,
    CertificationEntry,
    EducationEntry,
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

    entry_id: str | None = None

    text: str

    evidence_ids: list[str]

    requirement_ids: list[str]

    source_labels: list[str]

    validated: bool = True


class ResumeExperienceEntry(BaseModel):
    entry_id: str
    company: str
    title: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    bullets: list[
        ResumeBulletRecord
    ] = Field(default_factory=list)


class ResumeProjectEntry(BaseModel):
    entry_id: str
    name: str
    technologies: list[str] = Field(
        default_factory=list
    )
    date: str | None = None

    bullets: list[
        ResumeBulletRecord
    ] = Field(default_factory=list)


class TailoredResume(BaseModel):
    target_role: str | None = None
    company: str | None = None

    summary: list[
        ResumeBulletRecord
    ] = Field(default_factory=list)

    skills: list[str] = Field(
        default_factory=list
    )

    experience: list[
        ResumeExperienceEntry
    ] = Field(default_factory=list)

    projects: list[
        ResumeProjectEntry
    ] = Field(default_factory=list)

    education: list[
        EducationEntry
    ] = Field(default_factory=list)

    certifications: list[
        CertificationEntry
    ] = Field(default_factory=list)


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
    profile: CandidateProfile,
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
        profile,
    )

    summary: list[
        ResumeBulletRecord
    ] = []

    experience_bullets: dict[
        str,
        list[ResumeBulletRecord],
    ] = {}

    project_bullets: dict[
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

                        if (
                            label
                            not in labels
                        ):
                            labels.append(
                                label
                            )

                accepted = (
                    ResumeBulletRecord(
                        section=(
                            bullet.section
                        ),
                        entry_id=(
                            bullet.entry_id
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

        if accepted.section == "summary":
            summary.append(
                accepted
            )

        elif (
            accepted.section
            == "experience"
            and accepted.entry_id
        ):
            experience_bullets.setdefault(
                accepted.entry_id,
                [],
            ).append(
                accepted
            )

        elif (
            accepted.section
            == "projects"
            and accepted.entry_id
        ):
            project_bullets.setdefault(
                accepted.entry_id,
                [],
            ).append(
                accepted
            )

    experience = []

    for entry in profile.experience:
        bullets = experience_bullets.get(
            entry.entry_id,
            [],
        )

        if not bullets:
            continue

        experience.append(
            ResumeExperienceEntry(
                entry_id=entry.entry_id,
                company=entry.company,
                title=entry.title,
                location=entry.location,
                start_date=entry.start_date,
                end_date=entry.end_date,
                bullets=bullets,
            )
        )

    projects = []

    for entry in profile.projects:
        bullets = project_bullets.get(
            entry.entry_id,
            [],
        )

        if not bullets:
            continue

        projects.append(
            ResumeProjectEntry(
                entry_id=entry.entry_id,
                name=entry.name,
                technologies=(
                    entry.technologies
                ),
                date=entry.date,
                bullets=bullets,
            )
        )

    skills = (
        plan.skills_to_emphasize
        or profile.skills
    )

    return TailoredResume(
        target_role=(
            analysis.role_title
        ),
        company=analysis.company,
        summary=summary,
        skills=skills,
        experience=experience,
        projects=projects,
        education=profile.education,
        certifications=(
            profile.certifications
        ),
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
