from __future__ import annotations

import json

from pydantic import (
    BaseModel,
    ValidationError,
)

from local_llm import LLM, GenConfig

from generation.planner import (
    PlannedBullet,
    ResumeSection,
)


BULLET_RETRIES = 2
BULLET_MAX_NEW_TOKENS = 300
BULLET_REPAIR_MAX_NEW_TOKENS = 220


class BulletResponse(BaseModel):
    text: str


class GeneratedBullet(BaseModel):
    section: ResumeSection

    entry_id: str | None = None

    text: str

    evidence_ids: list[str]

    requirement_ids: list[str]


SYSTEM_PROMPT = """
You write evidence-grounded resume statements.

RULES:

1. Use ONLY the supplied evidence.
2. Never invent numbers, technologies, responsibilities, scale, impact, or outcomes.
3. Preserve factual metrics exactly.
4. Use strong action-oriented resume language without exaggeration.
5. Produce one concise sentence.
6. Do not use first-person pronouns.
7. Do not add information simply because it sounds impressive.
8. Return JSON only.
9. Do not include trailing commas.

Format:

{
  "text": "Built a Python REST API..."
}
""".strip()


REPAIR_SYSTEM_PROMPT = """
You repair JSON produced by a resume bullet generator.

RULES:

1. Preserve the exact factual meaning of the existing bullet.
2. Do not add technologies, metrics, responsibilities, outcomes, or claims.
3. Repair only JSON syntax or schema/type problems.
4. The only valid output shape is:
   {"text": "<one resume sentence>"}
5. Return JSON only.
6. Do not include trailing commas.
""".strip()


def normalize_text(
    text: str,
) -> str:
    text = text.strip()

    for prefix in (
        "- ",
        "• ",
        "* ",
    ):
        if text.startswith(prefix):
            text = text[
                len(prefix):
            ]

    return text.strip()


def _extract_json_text(
    raw: str,
) -> str:
    """
    Extract the outer JSON object from a model response.

    Handles plain JSON, fenced JSON, and small amounts of surrounding text.
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

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "Bullet generator output did not contain a JSON object."
        )

    return text[
        start:end + 1
    ]


def _remove_trailing_commas(
    text: str,
) -> str:
    """
    Remove commas immediately before } or ] while respecting JSON strings.

    Example:
        {"text": "Built an API.",}
    becomes:
        {"text": "Built an API."}

    This is intentionally narrow and does not attempt general JSON repair.
    """
    result: list[str] = []

    in_string = False
    escaped = False
    index = 0

    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)

            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1

            while (
                lookahead < len(text)
                and text[lookahead].isspace()
            ):
                lookahead += 1

            if (
                lookahead < len(text)
                and text[lookahead] in "}]"
            ):
                index += 1
                continue

        result.append(char)
        index += 1

    return "".join(result)


def parse_bullet_response(
    raw: str,
) -> BulletResponse:
    """
    Parse and validate a bullet response.

    A common small-model failure is a trailing comma. Repair that
    deterministically before asking the model to retry.
    """
    candidate = _extract_json_text(
        raw
    )

    try:
        parsed = json.loads(
            candidate
        )

    except json.JSONDecodeError as first_error:
        cleaned = _remove_trailing_commas(
            candidate
        )

        if cleaned == candidate:
            raise first_error

        parsed = json.loads(
            cleaned
        )

    return BulletResponse.model_validate(
        parsed
    )


def _repair_bullet_response(
    llm: LLM,
    raw: str,
    error: Exception,
) -> str:
    """
    Ask for a small, isolated JSON repair without resending the evidence payload.
    """
    if isinstance(
        error,
        ValidationError,
    ):
        instruction = (
            "The response is JSON but violates the required schema.\n"
            f"Validation error:\n{error}\n\n"
            "Repair only the schema/type problem."
        )
    else:
        instruction = (
            "The response has invalid JSON syntax.\n"
            f"Parsing error:\n{error}\n\n"
            "Repair only the JSON syntax."
        )

    return llm.generate(
        [
            {
                "role": "system",
                "content":
                    REPAIR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    instruction
                    + "\n\nBROKEN RESPONSE:\n"
                    + raw
                ),
            },
        ],
        config=GenConfig(
            max_new_tokens=(
                BULLET_REPAIR_MAX_NEW_TOKENS
            ),
            temperature=0.0,
        ),
    )


def generate_bullet(
    llm: LLM,
    plan: PlannedBullet,
    evidence_lookup: dict[
        str,
        dict,
    ],
    feedback: str | None = None,
) -> GeneratedBullet:
    evidence = []

    for evidence_id in (
        plan.evidence_ids
    ):
        card = evidence_lookup.get(
            evidence_id
        )

        if card:
            evidence.append(
                {
                    "evidence_id":
                        evidence_id,

                    "claim":
                        card.get("claim"),

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

    if not evidence:
        raise ValueError(
            "No valid evidence available "
            "for this bullet."
        )

    payload = {
        "section":
            plan.section,

        "entry_id":
            plan.entry_id,

        "objective":
            plan.objective,

        "evidence":
            evidence,
    }

    if feedback:
        payload[
            "previous_validation_feedback"
        ] = feedback

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
                BULLET_MAX_NEW_TOKENS
            ),
            temperature=0.0,
        ),
    )

    last_error: Exception | None = None

    for attempt in range(
        BULLET_RETRIES + 1
    ):
        try:
            response = parse_bullet_response(
                raw
            )
            break

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            last_error = error

            print(
                f"  Invalid bullet output "
                f"(attempt {attempt + 1}/"
                f"{BULLET_RETRIES + 1}): "
                f"{error}"
            )

            if attempt >= BULLET_RETRIES:
                raise RuntimeError(
                    "Bullet generation failed after "
                    f"{BULLET_RETRIES + 1} attempts: "
                    f"{last_error}"
                ) from error

            raw = _repair_bullet_response(
                llm,
                raw,
                error,
            )

    return GeneratedBullet(
        section=plan.section,
        entry_id=plan.entry_id,
        text=normalize_text(
            response.text
        ),
        evidence_ids=(
            plan.evidence_ids
        ),
        requirement_ids=(
            plan.requirement_ids
        ),
    )
