from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from llm import LLM, GenConfig

from jd.schemas import (
    JobDescriptionAnalysis,
    JobDescriptionExtraction,
    JobRequirement,
)


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

    for _ in range(
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

            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": raw,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Repair your previous "
                            "response and return "
                            "valid JSON only."
                        ),
                    },
                ]
            )

    raise RuntimeError(
        f"Requirement extraction failed: "
        f"{last_error}"
    )