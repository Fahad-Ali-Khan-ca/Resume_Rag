from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from config import CANDIDATE_PROFILE_FILE
from local_llm import GenConfig, LLM


PROFILE_MAX_NEW_TOKENS = 2048
PROFILE_RETRIES = 2


class ExperienceDraft(BaseModel):
    company: str
    title: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class ProjectDraft(BaseModel):
    name: str
    technologies: list[str] = Field(
        default_factory=list
    )
    date: str | None = None


class EducationDraft(BaseModel):
    institution: str
    degree: str
    field: str | None = None
    location: str | None = None
    graduation_date: str | None = None


class CertificationDraft(BaseModel):
    name: str
    issuer: str | None = None
    date: str | None = None


class CandidateProfileDraft(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None

    skills: list[str] = Field(
        default_factory=list
    )
    experience: list[
        ExperienceDraft
    ] = Field(default_factory=list)
    projects: list[
        ProjectDraft
    ] = Field(default_factory=list)
    education: list[
        EducationDraft
    ] = Field(default_factory=list)
    certifications: list[
        CertificationDraft
    ] = Field(default_factory=list)


class ExperienceEntry(ExperienceDraft):
    entry_id: str
    source_files: list[str] = Field(
        default_factory=list
    )


class ProjectEntry(ProjectDraft):
    entry_id: str
    source_files: list[str] = Field(
        default_factory=list
    )


class EducationEntry(EducationDraft):
    entry_id: str
    source_files: list[str] = Field(
        default_factory=list
    )


class CertificationEntry(CertificationDraft):
    entry_id: str
    source_files: list[str] = Field(
        default_factory=list
    )


class CandidateProfile(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None

    skills: list[str] = Field(
        default_factory=list
    )
    experience: list[
        ExperienceEntry
    ] = Field(default_factory=list)
    projects: list[
        ProjectEntry
    ] = Field(default_factory=list)
    education: list[
        EducationEntry
    ] = Field(default_factory=list)
    certifications: list[
        CertificationEntry
    ] = Field(default_factory=list)

    source_files: list[str] = Field(
        default_factory=list
    )


SYSTEM_PROMPT = """
You reconstruct the STRUCTURE of a candidate resume from unstructured source
text.

This is a fallback parser. Canonical ResumeForge Markdown is parsed by Python.
Extract only explicit information.

RULES:
1. Never invent employers, titles, dates, locations, projects, schools,
   degrees, certifications, technologies, links, or contact information.
2. Preserve names, titles, organizations, and dates as written when possible.
3. Use null when an optional scalar field is not explicitly present.
4. Use [] when an optional list has no supported values.
5. Do not turn responsibilities or accomplishments into structure fields.
6. Do not infer current employment unless dates explicitly indicate it.
7. Skills must be explicitly present in the source.
8. Return JSON only.

Required format:
{
  "name": null,
  "email": null,
  "phone": null,
  "location": null,
  "linkedin": null,
  "github": null,
  "skills": [],
  "experience": [
    {
      "company": "Company name",
      "title": "Job title",
      "location": null,
      "start_date": null,
      "end_date": null
    }
  ],
  "projects": [
    {
      "name": "Project name",
      "technologies": [],
      "date": null
    }
  ],
  "education": [
    {
      "institution": "School",
      "degree": "Degree",
      "field": null,
      "location": null,
      "graduation_date": null
    }
  ],
  "certifications": [
    {
      "name": "Certification",
      "issuer": null,
      "date": null
    }
  ]
}
""".strip()


def _document_value(
    document: Any,
    name: str,
    default=None,
):
    if isinstance(document, dict):
        return document.get(
            name,
            default,
        )

    return getattr(
        document,
        name,
        default,
    )


def _normalize_key(
    value: str | None,
) -> str:
    return (
        value or ""
    ).strip().casefold()


def _entry_id(
    prefix: str,
    *values: str | None,
) -> str:
    raw = "|".join(
        _normalize_key(value)
        for value in values
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return f"{prefix}_{digest[:12]}"


def _deduplicate_strings(
    values: list[str],
) -> list[str]:
    seen = set()
    result = []

    for value in values:
        if not isinstance(value, str):
            continue

        cleaned = value.strip()

        if not cleaned:
            continue

        key = cleaned.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result


def _extract_json_object(
    raw: str,
) -> dict:
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if (
            start == -1
            or end == -1
            or end <= start
        ):
            raise ValueError(
                "Model output did not contain a JSON object."
            )

        return json.loads(
            text[start:end + 1]
        )


def _normalize_profile_json(
    data: dict,
) -> dict:
    normalized = dict(data)

    for key in (
        "skills",
        "experience",
        "projects",
        "education",
        "certifications",
    ):
        if not isinstance(
            normalized.get(key),
            list,
        ):
            normalized[key] = []

    normalized["skills"] = [
        value.strip()
        for value in normalized["skills"]
        if isinstance(value, str)
        and value.strip()
    ]

    normalized["experience"] = [
        item
        for item in normalized[
            "experience"
        ]
        if isinstance(item, dict)
        and isinstance(
            item.get("company"),
            str,
        )
        and item["company"].strip()
        and isinstance(
            item.get("title"),
            str,
        )
        and item["title"].strip()
    ]

    normalized["projects"] = [
        item
        for item in normalized[
            "projects"
        ]
        if isinstance(item, dict)
        and isinstance(
            item.get("name"),
            str,
        )
        and item["name"].strip()
    ]

    normalized["education"] = [
        item
        for item in normalized[
            "education"
        ]
        if isinstance(item, dict)
        and isinstance(
            item.get("institution"),
            str,
        )
        and item[
            "institution"
        ].strip()
        and isinstance(
            item.get("degree"),
            str,
        )
        and item["degree"].strip()
    ]

    normalized[
        "certifications"
    ] = [
        item
        for item in normalized[
            "certifications"
        ]
        if isinstance(item, dict)
        and isinstance(
            item.get("name"),
            str,
        )
        and item["name"].strip()
    ]

    return normalized


def _build_source_prompt(
    source_file: str,
    documents: list[Any],
) -> str:
    pieces = []

    for document in sorted(
        documents,
        key=lambda item: (
            _document_value(
                item,
                "page",
                0,
            )
            or 0
        ),
    ):
        page = _document_value(
            document,
            "page",
        )
        text = _document_value(
            document,
            "text",
            "",
        )

        heading = (
            f"PAGE {page}"
            if page is not None
            else "DOCUMENT"
        )

        pieces.append(
            f"--- {heading} ---\n{text}"
        )

    return (
        "Extract candidate resume structure from this unstructured source.\n\n"
        f"SOURCE FILE:\n{source_file}\n\n"
        "SOURCE TEXT:\n"
        + "\n\n".join(pieces)
        + "\n\nReturn JSON only."
    )


def _extract_unstructured_profile(
    llm: LLM,
    source_file: str,
    documents: list[Any],
) -> CandidateProfileDraft:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": _build_source_prompt(
                source_file,
                documents,
            ),
        },
    ]

    config = GenConfig(
        max_new_tokens=(
            PROFILE_MAX_NEW_TOKENS
        ),
        temperature=0.0,
        top_p=1.0,
    )

    last_error: Exception | None = None

    for attempt in range(
        PROFILE_RETRIES + 1
    ):
        raw = llm.generate(
            messages,
            config=config,
        )

        try:
            parsed = (
                _normalize_profile_json(
                    _extract_json_object(
                        raw
                    )
                )
            )

            return (
                CandidateProfileDraft
                .model_validate(
                    parsed
                )
            )

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            last_error = error

            if attempt >= PROFILE_RETRIES:
                break

            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": raw,
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous response could not be validated.\n\n"
                            f"Error:\n{error}\n\n"
                            "Return the same source-supported information in "
                            "the required JSON structure. Do not invent "
                            "missing values. Return JSON only."
                        ),
                    },
                ]
            )

    raise RuntimeError(
        "Could not reconstruct candidate profile from "
        f"{source_file}: {last_error}"
    )


def _metadata(
    document: Any,
) -> dict:
    value = _document_value(
        document,
        "metadata",
        {},
    )

    return (
        value
        if isinstance(value, dict)
        else {}
    )


def extract_candidate_profile(
    llm: LLM,
    documents: list[Any],
) -> CandidateProfile:
    """
    Build candidate structure.

    Canonical Markdown documents are already deterministically structured by
    ingestion.canonical_parser, so their metadata is used directly and their
    entity IDs become resume entry IDs. Unstructured sources retain the LLM
    fallback for backwards compatibility/import workflows.
    """
    merged = CandidateProfile()

    source_files = _deduplicate_strings(
        [
            str(
                _document_value(
                    document,
                    "source_file",
                    "",
                )
            )
            for document in documents
            if _document_value(
                document,
                "source_file",
            )
        ]
    )
    merged.source_files = source_files

    experience_by_key: dict[
        tuple[str, ...],
        ExperienceEntry,
    ] = {}
    projects_by_key: dict[
        tuple[str, ...],
        ProjectEntry,
    ] = {}
    education_by_key: dict[
        tuple[str, ...],
        EducationEntry,
    ] = {}
    certifications_by_key: dict[
        tuple[str, ...],
        CertificationEntry,
    ] = {}
    all_skills: list[str] = []

    unstructured_by_source: dict[
        str,
        list[Any],
    ] = defaultdict(list)

    # Canonical structured documents ---------------------------------------
    for document in documents:
        entity_type = _document_value(
            document,
            "entity_type",
        )
        entity_id = _document_value(
            document,
            "entity_id",
        )
        source_file = str(
            _document_value(
                document,
                "source_file",
                "",
            )
        )

        if not entity_type or not entity_id:
            unstructured_by_source[
                source_file
            ].append(document)
            continue

        data = _metadata(document)

        if entity_type == "profile":
            for field_name in (
                "name",
                "email",
                "phone",
                "location",
                "linkedin",
                "github",
            ):
                if (
                    getattr(
                        merged,
                        field_name,
                    )
                    is None
                    and data.get(
                        field_name
                    )
                ):
                    setattr(
                        merged,
                        field_name,
                        data[field_name],
                    )

            all_skills.extend(
                data.get(
                    "skills",
                    [],
                )
            )
            continue

        if entity_type == "experience":
            item = ExperienceEntry(
                company=data["company"],
                title=data["title"],
                location=data.get(
                    "location"
                ),
                start_date=data.get(
                    "start_date"
                ),
                end_date=data.get(
                    "end_date"
                ),
                entry_id=str(entity_id),
                source_files=[
                    source_file
                ],
            )

            key = (
                _normalize_key(
                    item.company
                ),
                _normalize_key(
                    item.title
                ),
                _normalize_key(
                    item.start_date
                ),
                _normalize_key(
                    item.end_date
                ),
            )
            experience_by_key[key] = item
            continue

        if entity_type == "project":
            item = ProjectEntry(
                name=data["name"],
                technologies=(
                    _deduplicate_strings(
                        data.get(
                            "technologies",
                            [],
                        )
                    )
                ),
                date=data.get("date"),
                entry_id=str(entity_id),
                source_files=[
                    source_file
                ],
            )

            projects_by_key[
                (
                    _normalize_key(
                        item.name
                    ),
                )
            ] = item
            continue

        if entity_type == "education":
            item = EducationEntry(
                institution=(
                    data["institution"]
                ),
                degree=data["degree"],
                field=data.get("field"),
                location=data.get(
                    "location"
                ),
                graduation_date=(
                    data.get(
                        "graduation_date"
                    )
                ),
                entry_id=str(entity_id),
                source_files=[
                    source_file
                ],
            )

            education_by_key[
                (
                    _normalize_key(
                        item.institution
                    ),
                    _normalize_key(
                        item.degree
                    ),
                    _normalize_key(
                        item.field
                    ),
                    _normalize_key(
                        item.graduation_date
                    ),
                )
            ] = item
            continue

        if entity_type == "certification":
            item = CertificationEntry(
                name=data["name"],
                issuer=data.get(
                    "issuer"
                ),
                date=data.get("date"),
                entry_id=str(entity_id),
                source_files=[
                    source_file
                ],
            )

            certifications_by_key[
                (
                    _normalize_key(
                        item.name
                    ),
                    _normalize_key(
                        item.issuer
                    ),
                )
            ] = item

    # Unstructured fallback -------------------------------------------------
    for source_file, source_documents in (
        unstructured_by_source.items()
    ):
        if not source_documents:
            continue

        profile = (
            _extract_unstructured_profile(
                llm,
                source_file,
                source_documents,
            )
        )

        for field_name in (
            "name",
            "email",
            "phone",
            "location",
            "linkedin",
            "github",
        ):
            if (
                getattr(
                    merged,
                    field_name,
                )
                is None
                and getattr(
                    profile,
                    field_name,
                )
            ):
                setattr(
                    merged,
                    field_name,
                    getattr(
                        profile,
                        field_name,
                    ),
                )

        all_skills.extend(
            profile.skills
        )

        for draft in profile.experience:
            key = (
                _normalize_key(
                    draft.company
                ),
                _normalize_key(
                    draft.title
                ),
                _normalize_key(
                    draft.start_date
                ),
                _normalize_key(
                    draft.end_date
                ),
            )

            if key in experience_by_key:
                existing = (
                    experience_by_key[key]
                )
                if (
                    source_file
                    not in existing.source_files
                ):
                    existing.source_files.append(
                        source_file
                    )
                continue

            experience_by_key[key] = (
                ExperienceEntry(
                    **draft.model_dump(),
                    entry_id=_entry_id(
                        "exp",
                        draft.company,
                        draft.title,
                        draft.start_date,
                        draft.end_date,
                    ),
                    source_files=[
                        source_file
                    ],
                )
            )

        for draft in profile.projects:
            key = (
                _normalize_key(
                    draft.name
                ),
            )

            if key in projects_by_key:
                existing = (
                    projects_by_key[key]
                )
                existing.technologies = (
                    _deduplicate_strings(
                        existing.technologies
                        + draft.technologies
                    )
                )
                continue

            projects_by_key[key] = (
                ProjectEntry(
                    **draft.model_dump(),
                    entry_id=_entry_id(
                        "proj",
                        draft.name,
                    ),
                    source_files=[
                        source_file
                    ],
                )
            )

        for draft in profile.education:
            key = (
                _normalize_key(
                    draft.institution
                ),
                _normalize_key(
                    draft.degree
                ),
                _normalize_key(
                    draft.field
                ),
                _normalize_key(
                    draft.graduation_date
                ),
            )

            if key in education_by_key:
                continue

            education_by_key[key] = (
                EducationEntry(
                    **draft.model_dump(),
                    entry_id=_entry_id(
                        "edu",
                        draft.institution,
                        draft.degree,
                        draft.field,
                        draft.graduation_date,
                    ),
                    source_files=[
                        source_file
                    ],
                )
            )

        for draft in profile.certifications:
            key = (
                _normalize_key(
                    draft.name
                ),
                _normalize_key(
                    draft.issuer
                ),
            )

            if key in certifications_by_key:
                continue

            certifications_by_key[key] = (
                CertificationEntry(
                    **draft.model_dump(),
                    entry_id=_entry_id(
                        "cert",
                        draft.name,
                        draft.issuer,
                    ),
                    source_files=[
                        source_file
                    ],
                )
            )

    merged.skills = (
        _deduplicate_strings(
            all_skills
        )
    )
    merged.experience = list(
        experience_by_key.values()
    )
    merged.projects = list(
        projects_by_key.values()
    )
    merged.education = list(
        education_by_key.values()
    )
    merged.certifications = list(
        certifications_by_key.values()
    )

    return merged


def save_candidate_profile(
    profile: CandidateProfile,
    output_file: Path = (
        CANDIDATE_PROFILE_FILE
    ),
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        profile.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )
