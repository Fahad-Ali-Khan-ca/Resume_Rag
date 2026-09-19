from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CANONICAL_FILENAMES = {
    "profile.md",
    "experience.md",
    "projects.md",
}


@dataclass
class CanonicalEntity:
    """
    One semantic career entity parsed from the canonical Markdown corpus.

    A physical file can produce multiple entities. For example, experience.md
    produces one entity per employment entry.
    """

    entity_id: str
    entity_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        return self.entity_id


def is_canonical_markdown(path: Path) -> bool:
    return (
        path.suffix.casefold() == ".md"
        and path.name.casefold()
        in CANONICAL_FILENAMES
    )


def _strip_front_matter(text: str) -> str:
    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        return text.strip()

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(
                lines[index + 1:]
            ).strip()

    return text.strip()


def _heading(line: str) -> tuple[int, str] | None:
    match = re.match(
        r"^(#{1,6})\s+(.+?)\s*$",
        line.strip(),
    )

    if not match:
        return None

    return (
        len(match.group(1)),
        match.group(2).strip(),
    )


def _field(line: str) -> tuple[str, str] | None:
    match = re.match(
        r"^\s*([A-Za-z][A-Za-z0-9 _/&().+#-]*):\s*(.*?)\s*$",
        line,
    )

    if not match:
        return None

    return (
        match.group(1).strip().casefold(),
        match.group(2).strip(),
    )


def _bullet(line: str) -> str | None:
    stripped = line.strip()

    if not stripped.startswith(("- ", "* ")):
        return None

    value = stripped[2:].strip()

    if not value:
        return None

    if value.casefold() in {
        "none",
        "n/a",
        "na",
    }:
        return None

    return value


def _clean_scalar(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()

    if not cleaned:
        return None

    if cleaned.casefold() in {
        "none",
        "null",
        "n/a",
        "na",
    }:
        return None

    return cleaned


def _slug(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")

    return value or "entity"


def _entity_id(
    heading_text: str,
    prefix: str,
) -> str:
    """
    Preserve explicit canonical IDs when supplied.

    Examples:
        exp_digm_labs_2024 -> exp_digm_labs_2024
        Digm Labs -> exp_digm_labs
    """

    cleaned = _slug(
        heading_text
    )

    known_prefixes = {
        "exp": ("exp_", "experience_"),
        "proj": ("proj_", "project_"),
        "edu": ("edu_", "education_"),
        "cert": ("cert_", "certification_"),
    }

    if cleaned.startswith(
        known_prefixes.get(
            prefix,
            (f"{prefix}_",),
        )
    ):
        return cleaned

    return f"{prefix}_{cleaned}"


def _split_h2_blocks(
    text: str,
) -> list[tuple[str, list[str]]]:
    blocks: list[
        tuple[str, list[str]]
    ] = []

    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = _heading(line)

        if heading and heading[0] == 2:
            if current_title is not None:
                blocks.append(
                    (
                        current_title,
                        current_lines,
                    )
                )

            current_title = heading[1]
            current_lines = []
            continue

        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        blocks.append(
            (
                current_title,
                current_lines,
            )
        )

    return blocks


def _normalized_block_text(
    fields: list[tuple[str, str]],
    sections: list[tuple[str, list[str]]],
) -> str:
    lines: list[str] = []

    for name, value in fields:
        if value:
            lines.append(
                f"{name}: {value}"
            )

    for title, section_lines in sections:
        cleaned = [
            line.rstrip()
            for line in section_lines
            if line.strip()
        ]

        if not cleaned:
            continue

        if lines:
            lines.append("")

        lines.append(
            f"### {title}"
        )
        lines.extend(cleaned)

    return "\n".join(lines).strip()


def _parse_entity_block(
    lines: list[str],
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
]:
    fields: dict[str, str] = {}
    sections: dict[
        str,
        list[str],
    ] = {}

    current_section: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        heading = _heading(line)

        if heading and heading[0] == 3:
            current_section = (
                heading[1]
                .strip()
                .casefold()
            )
            sections.setdefault(
                current_section,
                [],
            )
            continue

        if current_section is None:
            field_value = _field(line)

            if field_value:
                key, value = field_value
                fields[key] = value

            continue

        sections[
            current_section
        ].append(line)

    return fields, sections


def _parse_experience(
    text: str,
) -> list[CanonicalEntity]:
    entities: list[CanonicalEntity] = []

    for heading_text, lines in (
        _split_h2_blocks(text)
    ):
        if heading_text.casefold() == "experience":
            continue

        fields, sections = (
            _parse_entity_block(lines)
        )

        company = _clean_scalar(
            fields.get("company")
        )
        title = _clean_scalar(
            fields.get("title")
        )

        if not company or not title:
            raise ValueError(
                "Each experience entry must include non-empty "
                f"Company and Title fields. Entry: {heading_text!r}"
            )

        entity_id = _entity_id(
            heading_text,
            "exp",
        )

        metadata = {
            "company": company,
            "title": title,
            "location": _clean_scalar(
                fields.get("location")
            ),
            "start_date": _clean_scalar(
                fields.get("start")
                or fields.get("start date")
            ),
            "end_date": _clean_scalar(
                fields.get("end")
                or fields.get("end date")
            ),
        }

        normalized_fields = [
            ("Company", metadata["company"]),
            ("Title", metadata["title"]),
            ("Location", metadata["location"]),
            ("Start", metadata["start_date"]),
            ("End", metadata["end_date"]),
        ]

        normalized_sections = [
            (
                title_name.title(),
                section_lines,
            )
            for title_name, section_lines
            in sections.items()
        ]

        entities.append(
            CanonicalEntity(
                entity_id=entity_id,
                entity_type="experience",
                text=_normalized_block_text(
                    normalized_fields,
                    normalized_sections,
                ),
                metadata=metadata,
            )
        )

    return entities


def _parse_projects(
    text: str,
) -> list[CanonicalEntity]:
    entities: list[CanonicalEntity] = []

    for heading_text, lines in (
        _split_h2_blocks(text)
    ):
        if heading_text.casefold() == "projects":
            continue

        fields, sections = (
            _parse_entity_block(lines)
        )

        name = _clean_scalar(
            fields.get("name")
        )

        if not name:
            raise ValueError(
                "Each project entry must include a non-empty Name field. "
                f"Entry: {heading_text!r}"
            )

        technologies: list[str] = []

        for line in sections.get(
            "technologies",
            [],
        ):
            value = _bullet(line)

            if (
                value
                and value
                not in technologies
            ):
                technologies.append(
                    value
                )

        entity_id = _entity_id(
            heading_text,
            "proj",
        )

        metadata = {
            "name": name,
            "project_type": _clean_scalar(
                fields.get("type")
            ),
            "repository": _clean_scalar(
                fields.get("repository")
            ),
            "date": _clean_scalar(
                fields.get("date")
                or fields.get("year")
            ),
            "technologies": technologies,
        }

        normalized_fields = [
            ("Name", metadata["name"]),
            ("Type", metadata["project_type"]),
            ("Repository", metadata["repository"]),
            ("Date", metadata["date"]),
        ]

        normalized_sections = [
            (
                title_name.title(),
                section_lines,
            )
            for title_name, section_lines
            in sections.items()
        ]

        entities.append(
            CanonicalEntity(
                entity_id=entity_id,
                entity_type="project",
                text=_normalized_block_text(
                    normalized_fields,
                    normalized_sections,
                ),
                metadata=metadata,
            )
        )

    return entities


def _section_range(
    lines: list[str],
    h2_title: str,
) -> list[str]:
    target = h2_title.casefold()
    collecting = False
    result: list[str] = []

    for line in lines:
        heading = _heading(line)

        if heading and heading[0] == 2:
            if collecting:
                break

            collecting = (
                heading[1]
                .casefold()
                == target
            )
            continue

        if collecting:
            result.append(line)

    return result


def _parse_profile(
    text: str,
) -> list[CanonicalEntity]:
    lines = text.splitlines()
    entities: list[CanonicalEntity] = []

    # Contact ---------------------------------------------------------------
    contact: dict[str, str | None] = {
        "name": None,
        "email": None,
        "phone": None,
        "location": None,
        "linkedin": None,
        "github": None,
    }

    for line in _section_range(
        lines,
        "Contact",
    ):
        field_value = _field(line)

        if not field_value:
            continue

        key, value = field_value

        if key in contact:
            contact[key] = (
                _clean_scalar(value)
            )

    # Summary ---------------------------------------------------------------
    summary_lines = [
        line.strip()
        for line in _section_range(
            lines,
            "Professional Summary",
        )
        if line.strip()
        and not _heading(line)
    ]

    summary = " ".join(
        summary_lines
    ).strip()

    # Skills ----------------------------------------------------------------
    skills: list[str] = []

    for line in _section_range(
        lines,
        "Skills",
    ):
        value = _bullet(line)

        if value and value not in skills:
            skills.append(value)

    profile_text_parts: list[str] = []

    if summary:
        profile_text_parts.extend(
            [
                "Professional Summary:",
                summary,
            ]
        )

    if skills:
        if profile_text_parts:
            profile_text_parts.append("")

        profile_text_parts.append(
            "Skills:"
        )
        profile_text_parts.extend(
            f"- {skill}"
            for skill in skills
        )

    entities.append(
        CanonicalEntity(
            entity_id="profile",
            entity_type="profile",
            text="\n".join(
                profile_text_parts
            ).strip(),
            metadata={
                **contact,
                "summary": (
                    summary or None
                ),
                "skills": skills,
            },
        )
    )

    # Education -------------------------------------------------------------
    education_lines = _section_range(
        lines,
        "Education",
    )

    current_institution: str | None = None
    current_lines: list[str] = []

    def flush_education() -> None:
        nonlocal current_institution
        nonlocal current_lines

        if not current_institution:
            return

        fields: dict[str, str] = {}
        achievements: list[str] = []
        current_subsection: str | None = None

        for raw_line in current_lines:
            heading = _heading(raw_line)

            if heading and heading[0] >= 4:
                current_subsection = (
                    heading[1]
                    .casefold()
                )
                continue

            field_value = _field(
                raw_line
            )

            if field_value:
                key, value = field_value

                if (
                    key
                    in {
                        "achievements",
                        "honours",
                        "honors",
                    }
                    and not value.strip()
                ):
                    current_subsection = key
                    continue

                fields[key] = value
                continue

            value = _bullet(raw_line)

            if (
                value
                and current_subsection
                in {
                    "achievements",
                    "honours",
                    "honors",
                }
            ):
                achievements.append(value)

        degree = _clean_scalar(
            fields.get("degree")
        )

        if not degree:
            raise ValueError(
                "Each education entry must include a non-empty Degree field. "
                f"Institution: {current_institution!r}"
            )

        metadata = {
            "institution": current_institution,
            "degree": degree,
            "field": _clean_scalar(
                fields.get("field")
            ),
            "location": _clean_scalar(
                fields.get("location")
            ),
            "graduation_date": _clean_scalar(
                fields.get("graduation")
                or fields.get(
                    "graduation date"
                )
            ),
            "achievements": achievements,
        }

        text_lines = [
            f"Institution: {metadata['institution']}",
            f"Degree: {metadata['degree']}",
        ]

        for label, key in (
            ("Field", "field"),
            ("Location", "location"),
            (
                "Graduation",
                "graduation_date",
            ),
        ):
            if metadata[key]:
                text_lines.append(
                    f"{label}: {metadata[key]}"
                )

        if achievements:
            text_lines.extend(
                [
                    "",
                    "Achievements:",
                    *[
                        f"- {item}"
                        for item in achievements
                    ],
                ]
            )

        entities.append(
            CanonicalEntity(
                entity_id=_entity_id(
                    current_institution,
                    "edu",
                ),
                entity_type="education",
                text="\n".join(
                    text_lines
                ),
                metadata=metadata,
            )
        )

        current_institution = None
        current_lines = []

    for line in education_lines:
        heading = _heading(line)

        if heading and heading[0] == 3:
            flush_education()
            current_institution = (
                heading[1].strip()
            )
            current_lines = []
            continue

        if current_institution:
            current_lines.append(line)

    flush_education()

    # Certifications --------------------------------------------------------
    certification_lines = _section_range(
        lines,
        "Certifications",
    )

    for line in certification_lines:
        value = _bullet(line)

        if not value:
            continue

        entities.append(
            CanonicalEntity(
                entity_id=_entity_id(
                    value,
                    "cert",
                ),
                entity_type="certification",
                text=f"Certification: {value}",
                metadata={
                    "name": value,
                    "issuer": None,
                    "date": None,
                },
            )
        )

    return entities


def parse_canonical_markdown(
    path: Path,
) -> list[CanonicalEntity]:
    if not is_canonical_markdown(path):
        raise ValueError(
            f"Not a canonical Markdown file: {path}"
        )

    text = _strip_front_matter(
        path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )

    filename = path.name.casefold()

    if filename == "experience.md":
        return _parse_experience(text)

    if filename == "projects.md":
        return _parse_projects(text)

    if filename == "profile.md":
        return _parse_profile(text)

    raise ValueError(
        f"Unsupported canonical file: {path.name}"
    )
