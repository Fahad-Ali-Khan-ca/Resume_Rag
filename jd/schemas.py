from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


RequirementCategory = Literal[
    "technical",
    "experience",
    "education",
    "certification",
    "soft_skill",
    "location",
    "work_authorization",
    "other",
]

RequirementImportance = Literal[
    "required",
    "preferred",
    "nice_to_have",
    "unknown",
]


def normalize_years_experience(
    value,
) -> float | None:
    """
    Normalize model output to the minimum explicitly stated years.

    Examples:
        5           -> 5.0
        "5"         -> 5.0
        "5+"        -> 5.0
        "2-3"       -> 2.0
        "2–3 years" -> 2.0
        "2 to 3"    -> 2.0
        "at least 4"-> 4.0

    The full wording remains preserved in requirement.text. This field is a
    normalized numeric value for ranking/filtering only.
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            "years_experience cannot be a boolean"
        )

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    if not isinstance(
        value,
        str,
    ):
        return value

    text = value.strip().lower()

    if not text or text in {
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
    }:
        return None

    # The normalized meaning is the minimum stated number of years.
    #
    # This handles:
    #   "2-3"
    #   "2–3 years"
    #   "2 to 3 years"
    #   "5+"
    #   "at least 5 years"
    #   "minimum 5 years"
    match = re.search(
        r"(?<!\d)(\d+(?:\.\d+)?)",
        text,
    )

    if not match:
        raise ValueError(
            "years_experience must contain a numeric year value"
        )

    return float(
        match.group(1)
    )


class ExtractedRequirement(BaseModel):
    """
    Raw requirement returned by the LLM.
    """

    text: str = Field(
        min_length=3
    )

    category: RequirementCategory

    importance: RequirementImportance = (
        "unknown"
    )

    skills: list[str] = Field(
        default_factory=list
    )

    years_experience: float | None = None

    @field_validator(
        "years_experience",
        mode="before",
    )
    @classmethod
    def normalize_years(
        cls,
        value,
    ):
        return normalize_years_experience(
            value
        )


class JobDescriptionExtraction(BaseModel):
    """
    Raw structured output expected from the model.
    """

    role_title: str | None = None
    company: str | None = None

    requirements: list[
        ExtractedRequirement
    ] = Field(
        default_factory=list
    )


class JobRequirement(BaseModel):
    """
    Normalized requirement used by ResumeForge.

    years_experience stores the minimum explicitly stated number of years.
    The original wording/range is preserved in `text`.
    """

    requirement_id: str

    text: str

    category: RequirementCategory

    importance: RequirementImportance

    skills: list[str]

    years_experience: float | None = None

    @field_validator(
        "years_experience",
        mode="before",
    )
    @classmethod
    def normalize_years(
        cls,
        value,
    ):
        return normalize_years_experience(
            value
        )

    search_query: str


class JobDescriptionAnalysis(BaseModel):
    """
    Final parsed representation of a job description.
    """

    role_title: str | None = None
    company: str | None = None

    requirements: list[
        JobRequirement
    ] = Field(
        default_factory=list
    )
