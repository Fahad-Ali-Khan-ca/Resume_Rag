from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from local_llm import LLM, GenConfig

from jd.schemas import (
    JobDescriptionAnalysis,
)

from matching.evidence_map import (
    EvidenceMap,
)

from generation.candidate_profile import (
    CandidateProfile,
)


PLANNER_RETRIES = 2
PLANNER_MAX_NEW_TOKENS = 1400

# The planner should never receive the entire RAG corpus. Retrieval/matching
# chooses priority evidence; this cap leaves room for the system prompt,
# candidate structure, job requirements, and generated JSON inside an 8k
# context model such as Gemma 2 2B.
PLANNER_MAX_EVIDENCE = 28


ResumeSection = Literal[
    "summary",
    "experience",
    "projects",
]


class DraftPlannedBullet(BaseModel):
    section: ResumeSection

    entry_id: str | None = None

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

You receive four things:

1. Candidate profile structure.
2. Extracted evidence cards.
3. Job requirements.
4. Requirement-to-evidence matches.

The candidate profile defines the resume structure.
Evidence cards define what factual claims may be written.
Requirement matches determine PRIORITY, not whether basic resume structure exists.

RULES:

1. Use only supplied evidence IDs for factual resume statements.
2. Never invent employers, titles, projects, technologies, dates, metrics,
   responsibilities, or outcomes.
3. Prioritize required job requirements and stronger evidence matches.
4. You MAY use evidence that was not matched to a job requirement when it is
   useful for producing a coherent and complete resume.
5. Avoid redundant bullets.
6. Prefer measurable and technically specific evidence.
7. Summary bullets must use section="summary" and entry_id=null.
8. Experience bullets must use section="experience" and a valid experience
   entry_id from candidate_profile.experience.
9. Every evidence card used for an experience bullet must have entity_id equal
   to that experience entry_id and entity_type="experience".
10. Project bullets must use section="projects" and a valid project entry_id
    from candidate_profile.projects.
11. Every evidence card used for a project bullet must have entity_id equal to
    that project entry_id and entity_type="project".
12. Never move evidence from one employer/project to another, even when the
    technologies or responsibilities look similar.
13. Summary bullets may combine evidence from multiple entities.
14. Do not create education or certification bullets. Those sections are
    preserved directly from the candidate profile.
15. Every planned bullet must reference at least one supplied evidence ID.
16. Use requirement_ids only when the bullet directly supports those
    requirements.
17. Return JSON only.

Format:

{
  "summary_focus": [
    "Python backend development"
  ],
  "skills_to_emphasize": [
    "Python",
    "SQL"
  ],
  "bullets": [
    {
      "section": "experience",
      "entry_id": "exp_x",
      "objective": "Demonstrate Python API development",
      "requirement_ids": ["req_x"],
      "evidence_ids": ["ev_x"]
    },
    {
      "section": "summary",
      "entry_id": null,
      "objective": "Summarize grounded backend experience",
      "requirement_ids": ["req_x"],
      "evidence_ids": ["ev_y"]
    }
  ]
}
""".strip()



def extract_json_object(
    raw: str,
) -> dict:
    """
    Parse a JSON object from an LLM response.

    Accepts plain JSON, fenced JSON, or a JSON object surrounded by a small
    amount of explanatory text. Syntax errors are allowed to propagate so the
    planner can ask the model for a focused repair.
    """

    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    try:
        return json.loads(
            text
        )

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "Planner output did not contain a JSON object."
        )

    return json.loads(
        text[start:end + 1]
    )


def parse_planner_response(
    llm: LLM,
    raw_response: str,
    retries: int = PLANNER_RETRIES,
) -> PlannerResponse:
    """
    Parse and validate planner output, repairing malformed model JSON when
    necessary.

    Repairs use a fresh conversation containing only the invalid response and
    parser/schema error. This avoids duplicating the large planning payload in
    the model context.
    """

    current_response = raw_response
    last_error: Exception | None = None

    for attempt in range(
        retries + 1
    ):
        try:
            parsed = extract_json_object(
                current_response
            )

            return PlannerResponse.model_validate(
                parsed
            )

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            last_error = error

            print(
                f"  Invalid planner output "
                f"(attempt {attempt + 1}/{retries + 1}): "
                f"{error}"
            )

            if attempt >= retries:
                break

            if isinstance(
                error,
                ValidationError,
            ):
                repair_instruction = (
                    "The JSON is parseable but does not match the required "
                    "planner schema. Repair only schema/type problems. "
                    "Preserve the same factual intent and existing IDs. "
                    "Do not invent employers, projects, evidence IDs, "
                    "requirement IDs, dates, skills, or claims."
                )
            else:
                repair_instruction = (
                    "Repair only the JSON syntax. Preserve the same keys, "
                    "values, arrays, bullet plans, and IDs. Do not add or "
                    "reinterpret factual content."
                )

            current_response = llm.generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "You repair structured JSON produced by a resume "
                            "planner. Return one complete valid JSON object "
                            "only. Do not use Markdown fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{repair_instruction}\n\n"
                            f"Parser/schema error:\n{error}\n\n"
                            "Invalid planner response:\n"
                            f"{current_response}"
                        ),
                    },
                ],
                config=GenConfig(
                    max_new_tokens=PLANNER_MAX_NEW_TOKENS,
                    temperature=0.0,
                ),
            )

    raise RuntimeError(
        f"Could not obtain valid planner JSON: {last_error}"
    )

def make_plan_id(
    bullet: DraftPlannedBullet,
) -> str:
    raw = (
        bullet.section
        + (bullet.entry_id or "")
        + bullet.objective
        + "|".join(
            bullet.evidence_ids
        )
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return f"plan_{digest[:12]}"


def _compact_profile(
    profile: CandidateProfile,
) -> dict:
    """
    Keep only structural information the planner needs.

    Provenance fields such as source_files are intentionally omitted because
    they consume context without helping the planner choose resume content.
    """
    return {
        "name": profile.name,
        "skills": profile.skills,
        "experience": [
            {
                "entry_id": entry.entry_id,
                "company": entry.company,
                "title": entry.title,
                "location": entry.location,
                "start_date": entry.start_date,
                "end_date": entry.end_date,
            }
            for entry in profile.experience
        ],
        "projects": [
            {
                "entry_id": entry.entry_id,
                "name": entry.name,
                "technologies": entry.technologies,
                "date": entry.date,
            }
            for entry in profile.projects
        ],
        "education": [
            {
                "entry_id": entry.entry_id,
                "institution": entry.institution,
                "degree": entry.degree,
                "field": entry.field,
                "graduation_date": entry.graduation_date,
            }
            for entry in profile.education
        ],
        "certifications": [
            {
                "entry_id": entry.entry_id,
                "name": entry.name,
                "issuer": entry.issuer,
                "date": entry.date,
            }
            for entry in profile.certifications
        ],
    }


def _compact_requirements(
    analysis: JobDescriptionAnalysis,
) -> list[dict]:
    return [
        {
            "requirement_id": item.requirement_id,
            "text": item.text,
            "importance": item.importance,
            "skills": item.skills,
            "years_experience": item.years_experience,
        }
        for item in analysis.requirements
    ]


def _compact_matches(
    evidence_map: EvidenceMap,
) -> list[dict]:
    return [
        {
            "requirement_id": item.requirement_id,
            "match_strength": item.match_strength,
            "evidence_ids": item.evidence_ids,
        }
        for item in evidence_map.requirements
    ]


def _evidence_score(
    evidence_id: str,
    card: dict,
    matched_strength: dict[str, int],
) -> tuple:
    """
    Deterministically rank evidence for the planner.

    Matched evidence wins first. Then prefer concrete experience/project/
    achievement cards, cards with metrics, and canonical Markdown sources.
    """
    category_rank = {
        "achievement": 5,
        "experience": 4,
        "project": 4,
        "responsibility": 3,
        "certification": 2,
        "education": 2,
        "skill": 1,
        "other": 0,
    }

    source_file = str(
        card.get("source_file") or ""
    )
    source_name = Path(
        source_file
    ).name.casefold()

    canonical_bonus = (
        2
        if source_name
        in {
            "profile.md",
            "experience.md",
            "projects.md",
        }
        else 0
    )

    metrics = card.get(
        "metrics",
        [],
    )

    metric_bonus = (
        2
        if isinstance(metrics, list)
        and metrics
        else 0
    )

    skills = card.get(
        "skills",
        [],
    )

    skill_bonus = min(
        len(skills)
        if isinstance(skills, list)
        else 0,
        3,
    )

    return (
        matched_strength.get(
            evidence_id,
            0,
        ),
        canonical_bonus,
        category_rank.get(
            str(
                card.get("category")
                or ""
            ),
            0,
        ),
        metric_bonus,
        skill_bonus,
        evidence_id,
    )


def _select_planner_evidence(
    evidence_lookup: dict[
        str,
        dict,
    ],
    evidence_map: EvidenceMap,
    profile: CandidateProfile,
    limit: int = PLANNER_MAX_EVIDENCE,
) -> list[dict]:
    """
    Select a bounded, entity-aware evidence subset for one planning prompt.

    Priority order:
    1. Evidence explicitly matched to the JD.
    2. Minimum coverage for every experience/project entity.
    3. Highest-value remaining evidence until the context cap is reached.
    """
    matched_strength: dict[
        str,
        int,
    ] = {}

    strength_value = {
        "strong": 100,
        "partial": 70,
        "none": 0,
    }

    for mapped in evidence_map.requirements:
        strength = strength_value.get(
            str(
                mapped.match_strength
            ).casefold(),
            0,
        )

        for evidence_id in mapped.evidence_ids:
            matched_strength[
                evidence_id
            ] = max(
                matched_strength.get(
                    evidence_id,
                    0,
                ),
                strength,
            )

    ranked = sorted(
        evidence_lookup.items(),
        key=lambda item: _evidence_score(
            item[0],
            item[1],
            matched_strength,
        ),
        reverse=True,
    )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    seen_claims: set[
        tuple[str, str]
    ] = set()

    def add_card(
        evidence_id: str,
        card: dict,
    ) -> bool:
        if len(selected) >= limit:
            return False

        if evidence_id in selected_ids:
            return False

        claim = str(
            card.get("claim")
            or ""
        ).strip()

        if not claim:
            return False

        claim_key = (
            str(
                card.get("entity_id")
                or ""
            ).casefold(),
            claim.casefold(),
        )

        if claim_key in seen_claims:
            return False

        selected_ids.add(
            evidence_id
        )
        seen_claims.add(
            claim_key
        )

        selected.append(
            {
                "evidence_id":
                    evidence_id,
                "claim": claim,
                "category":
                    card.get(
                        "category"
                    ),
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
                "entity_id":
                    card.get(
                        "entity_id"
                    ),
                "entity_type":
                    card.get(
                        "entity_type"
                    ),
            }
        )

        return True

    # First keep evidence already selected by the JD matching stage.
    for evidence_id, card in ranked:
        if matched_strength.get(
            evidence_id,
            0,
        ) > 0:
            add_card(
                evidence_id,
                card,
            )

    # Then guarantee each structured job/project has some evidence available to
    # the planner. This prevents global ranking from starving one resume entry.
    entity_ids = [
        entry.entry_id
        for entry in profile.experience
    ] + [
        entry.entry_id
        for entry in profile.projects
    ]

    for entity_id in entity_ids:
        existing_count = sum(
            1
            for item in selected
            if item.get("entity_id")
            == entity_id
        )

        needed = max(
            0,
            3 - existing_count,
        )

        if needed == 0:
            continue

        for evidence_id, card in ranked:
            if (
                card.get("entity_id")
                != entity_id
            ):
                continue

            if add_card(
                evidence_id,
                card,
            ):
                needed -= 1

            if needed <= 0:
                break

    # Fill remaining room with the strongest global evidence.
    for evidence_id, card in ranked:
        if len(selected) >= limit:
            break

        add_card(
            evidence_id,
            card,
        )

    return selected


def plan_resume(
    llm: LLM,
    analysis: JobDescriptionAnalysis,
    evidence_map: EvidenceMap,
    evidence_lookup: dict[
        str,
        dict,
    ],
    profile: CandidateProfile,
) -> ResumePlan:
    """
    Plan a complete structured resume using a bounded context.

    The RAG corpus can grow arbitrarily large. Retrieval/matching operates over
    the full evidence store, while the planner sees only the highest-value
    subset plus compact candidate structure and job requirements.
    """

    evidence_payload = (
        _select_planner_evidence(
            evidence_lookup,
            evidence_map,
            profile,
        )
    )

    supplied_evidence_ids = {
        item["evidence_id"]
        for item in evidence_payload
    }

    supported_skills = set()

    for item in evidence_payload:
        supported_skills.update(
            skill
            for skill in item.get(
                "skills",
                [],
            )
            if isinstance(
                skill,
                str,
            )
            and skill.strip()
        )

    payload = {
        "target_role":
            analysis.role_title,

        "company":
            analysis.company,

        "requirements":
            _compact_requirements(
                analysis
            ),

        "requirement_matches":
            _compact_matches(
                evidence_map
            ),

        "candidate_profile":
            _compact_profile(
                profile
            ),

        "evidence":
            evidence_payload,
    }

    print(
        f"Planner evidence: "
        f"{len(evidence_payload)}/"
        f"{len(evidence_lookup)} cards"
    )

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
                        separators=(
                            ",",
                            ":",
                        ),
                        ensure_ascii=False,
                    ),
            },
        ],
        config=GenConfig(
            max_new_tokens=(
                PLANNER_MAX_NEW_TOKENS
            ),
            temperature=0.0,
        ),
    )

    response = parse_planner_response(
        llm,
        raw,
    )

    experience_ids = {
        entry.entry_id
        for entry
        in profile.experience
    }

    project_ids = {
        entry.entry_id
        for entry
        in profile.projects
    }

    requirement_ids = {
        requirement.requirement_id
        for requirement
        in analysis.requirements
    }

    bullets = []

    for bullet in response.bullets:
        valid_evidence = [
            evidence_id
            for evidence_id
            in bullet.evidence_ids
            if evidence_id
            in supplied_evidence_ids
        ]

        if not valid_evidence:
            continue

        valid_requirements = [
            requirement_id
            for requirement_id
            in bullet.requirement_ids
            if requirement_id
            in requirement_ids
        ]

        if bullet.section == "summary":
            bullet.entry_id = None

        elif bullet.section == "experience":
            if (
                bullet.entry_id
                not in experience_ids
            ):
                continue

            valid_evidence = [
                evidence_id
                for evidence_id
                in valid_evidence
                if (
                    evidence_lookup.get(
                        evidence_id,
                        {},
                    ).get("entity_id")
                    == bullet.entry_id
                    and evidence_lookup.get(
                        evidence_id,
                        {},
                    ).get("entity_type")
                    == "experience"
                )
            ]

            if not valid_evidence:
                continue

        elif bullet.section == "projects":
            if (
                bullet.entry_id
                not in project_ids
            ):
                continue

            valid_evidence = [
                evidence_id
                for evidence_id
                in valid_evidence
                if (
                    evidence_lookup.get(
                        evidence_id,
                        {},
                    ).get("entity_id")
                    == bullet.entry_id
                    and evidence_lookup.get(
                        evidence_id,
                        {},
                    ).get("entity_type")
                    == "project"
                )
            ]

            if not valid_evidence:
                continue

        bullet.evidence_ids = (
            valid_evidence
        )
        bullet.requirement_ids = (
            valid_requirements
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

    supported_skill_lookup = {
        skill.casefold(): skill
        for skill
        in supported_skills
    }

    valid_skills = []

    for skill in (
        response.skills_to_emphasize
    ):
        if not isinstance(
            skill,
            str,
        ):
            continue

        matched = (
            supported_skill_lookup.get(
                skill.casefold()
            )
        )

        if (
            matched
            and matched
            not in valid_skills
        ):
            valid_skills.append(
                matched
            )

    if not valid_skills:
        for skill in profile.skills:
            if not isinstance(
                skill,
                str,
            ):
                continue

            matched = (
                supported_skill_lookup.get(
                    skill.casefold()
                )
            )

            if (
                matched
                and matched
                not in valid_skills
            ):
                valid_skills.append(
                    matched
                )

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
