from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class ExtractedRequirement(BaseModel):
    """
    Raw requirement returned by the LLM.
    """

    text: str = Field(min_length=3)

    category: RequirementCategory

    importance: RequirementImportance = "unknown"

    skills: list[str] = Field(
        default_factory=list
    )

    years_experience: float | None = None


class JobDescriptionExtraction(BaseModel):
    """
    Raw structured output expected from Gemma.
    """

    role_title: str | None = None
    company: str | None = None

    requirements: list[
        ExtractedRequirement
    ] = Field(default_factory=list)


class JobRequirement(BaseModel):
    """
    Normalized requirement used by ResumeForge.
    """

    requirement_id: str

    text: str

    category: RequirementCategory

    importance: RequirementImportance

    skills: list[str]

    years_experience: float | None = None

    search_query: str


class JobDescriptionAnalysis(BaseModel):
    """
    Final parsed representation of a job description.
    """

    role_title: str | None = None
    company: str | None = None

    requirements: list[
        JobRequirement
    ] = Field(default_factory=list)