from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from local_llm import LLM, GenConfig

from jd.schemas import (
    JobDescriptionAnalysis,
    JobDescriptionExtraction,
    JobRequirement,
)


VALID_CATEGORIES = {
    "technical",
    "experience",
    "education",
    "certification",
    "soft_skill",
    "location",
    "work_authorization",
    "other",
}

VALID_IMPORTANCE = {
    "required",
    "preferred",
    "nice_to_have",
    "unknown",
}


SYSTEM_PROMPT = """
You are the job-description analysis component of ResumeForge.

Extract explicit hiring requirements from the supplied job description.

RULES:

1. Do not invent requirements.
2. Separate distinct requirements.
3. Preserve technologies and skills exactly when possible.
4. Distinguish required requirements from preferred/nice-to-have ones.
5. Extract years of experience only when explicitly stated.
6. Soft skills should be separate from technical skills.
7. Ignore company marketing language and generic benefits.
8. Return valid JSON only.
9. "category" and "importance" are different fields.
10. Never put required, preferred, nice_to_have, or unknown in "category".
11. Never put technical, experience, education, certification, soft_skill,
    location, work_authorization, or other in "importance".

Allowed categories:

technical
experience
education
certification
soft_skill
location
work_authorization
other

Allowed importance values:

required
preferred
nice_to_have
unknown

Required format:

{
  "role_title": "Software Engineer",
  "company": "Example Corp",
  "requirements": [
    {
      "text": "Experience building REST APIs with Python",
      "category": "technical",
      "importance": "required",
      "skills": ["Python", "REST APIs"],
      "years_experience": null
    }
  ]
}
"""


def extract_json_object(
    text: str,
) -> dict:

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(lines)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "No JSON object found."
            )

        return json.loads(
            text[start:end + 1]
        )


def _infer_category(
    requirement: dict,
) -> str:
    """
    Infer only the broad schema category from already-extracted fields.

    This does not add a requirement or invent a skill. It is used only when
    the model accidentally places an importance enum (for example
    'nice_to_have') in the category field.
    """

    if requirement.get("years_experience") is not None:
        return "experience"

    skills = requirement.get("skills")

    if isinstance(skills, list) and any(
        isinstance(skill, str) and skill.strip()
        for skill in skills
    ):
        return "technical"

    text = str(
        requirement.get("text", "")
    ).lower()

    if any(
        token in text
        for token in (
            "degree",
            "bachelor",
            "master",
            "phd",
            "education",
        )
    ):
        return "education"

    if any(
        token in text
        for token in (
            "certification",
            "certified",
            "certificate",
        )
    ):
        return "certification"

    if any(
        token in text
        for token in (
            "work authorization",
            "authorized to work",
            "sponsorship",
        )
    ):
        return "work_authorization"

    if any(
        token in text
        for token in (
            "location",
            "onsite",
            "on-site",
            "hybrid",
            "remote",
        )
    ):
        return "location"

    return "other"


def normalize_requirement_enums(
    parsed: dict,
) -> dict:
    """
    Repair only obvious category/importance enum placement mistakes.

    Example:
        category="nice_to_have"
        importance="unknown"

    becomes:
        category="technical"  # if skills are present
        importance="nice_to_have"

    Unknown structural/schema errors are intentionally left for Pydantic.
    """

    requirements = parsed.get(
        "requirements"
    )

    if not isinstance(
        requirements,
        list,
    ):
        return parsed

    for requirement in requirements:

        if not isinstance(
            requirement,
            dict,
        ):
            continue

        category = requirement.get(
            "category"
        )
        importance = requirement.get(
            "importance"
        )

        if category in VALID_IMPORTANCE:

            if (
                importance is None
                or importance == "unknown"
                or importance not in VALID_IMPORTANCE
            ):
                requirement[
                    "importance"
                ] = category

            requirement[
                "category"
            ] = _infer_category(
                requirement
            )

        importance = requirement.get(
            "importance"
        )

        if importance in VALID_CATEGORIES:

            if (
                requirement.get("category")
                not in VALID_CATEGORIES
            ):
                requirement[
                    "category"
                ] = importance

            requirement[
                "importance"
            ] = "unknown"

    return parsed


def make_requirement_id(
    text: str,
) -> str:

    digest = hashlib.sha256(
        text.strip()
        .lower()
        .encode("utf-8")
    ).hexdigest()

    return f"req_{digest[:12]}"


def deduplicate(
    values: list[str],
) -> list[str]:

    seen = set()
    result = []

    for value in values:

        cleaned = value.strip()

        if not cleaned:
            continue

        key = cleaned.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result


def build_search_query(
    text: str,
    skills: list[str],
) -> str:

    if not skills:
        return text

    return (
        f"{text}. "
        f"Skills: {', '.join(skills)}"
    )


def _repair_prompt(
    error: Exception,
) -> str:

    if isinstance(
        error,
        ValidationError,
    ):
        return (
            "Your previous response is valid JSON but does not match "
            "the required schema.\n\n"
            f"Validation error:\n{error}\n\n"
            "Repair only the schema/type problems.\n"
            "Do not invent, remove, or reinterpret job requirements.\n"
            "Remember:\n"
            "- category must be one of: technical, experience, education, "
            "certification, soft_skill, location, work_authorization, other\n"
            "- importance must be one of: required, preferred, "
            "nice_to_have, unknown\n"
            "- nice_to_have is an importance value, never a category\n"
            "Return the complete corrected JSON object only."
        )

    return (
        "Your previous response is not valid JSON.\n\n"
        f"Parsing error:\n{error}\n\n"
        "Repair only the JSON syntax.\n"
        "Do not invent, remove, or reinterpret job requirements.\n"
        "Return the complete corrected JSON object only."
    )


def extract_requirements(
    llm: LLM,
    job_description: str,
    retries: int = 2,
) -> JobDescriptionAnalysis:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Analyze this job description:\n\n"
                + job_description
            ),
        },
    ]

    config = GenConfig(
        max_new_tokens=1800,
        temperature=0.0,
        top_p=1.0,
    )

    last_error = None

    for attempt in range(
        retries + 1
    ):

        raw = llm.generate(
            messages,
            config=config,
        )

        try:

            parsed = extract_json_object(
                raw
            )

            parsed = normalize_requirement_enums(
                parsed
            )

            extraction = (
                JobDescriptionExtraction
                .model_validate(parsed)
            )

            requirements = []

            for item in (
                extraction.requirements
            ):

                skills = deduplicate(
                    item.skills
                )

                requirements.append(
                    JobRequirement(
                        requirement_id=(
                            make_requirement_id(
                                item.text
                            )
                        ),
                        text=item.text.strip(),
                        category=item.category,
                        importance=(
                            item.importance
                        ),
                        skills=skills,
                        years_experience=(
                            item.years_experience
                        ),
                        search_query=(
                            build_search_query(
                                item.text,
                                skills,
                            )
                        ),
                    )
                )

            return JobDescriptionAnalysis(
                role_title=(
                    extraction.role_title
                ),
                company=(
                    extraction.company
                ),
                requirements=requirements,
            )

        except (
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ) as error:

            last_error = error

            print(
                f"  Invalid JD analysis "
                f"(attempt {attempt + 1}/{retries + 1}): "
                f"{error}"
            )

            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": raw,
                    },
                    {
                        "role": "user",
                        "content": _repair_prompt(
                            error
                        ),
                    },
                ]
            )

    raise RuntimeError(
        f"Requirement extraction failed: "
        f"{last_error}"
    )
