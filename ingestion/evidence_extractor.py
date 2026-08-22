from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from llm import LLM, GenConfig


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

CHUNKS_FILE = Path("corpus/processed/chunks.jsonl")
EVIDENCE_FILE = Path("corpus/processed/evidence.jsonl")
STATE_FILE = Path("corpus/processed/evidence_state.json")


# --------------------------------------------------------------------------- #
# Extraction schema
# --------------------------------------------------------------------------- #

EvidenceCategory = Literal[
    "skill",
    "project",
    "experience",
    "achievement",
    "education",
    "certification",
    "responsibility",
    "other",
]


class ExtractedMetric(BaseModel):
    """
    A measurable result explicitly present in the source text.

    Example:
        name="precision"
        value="76.9%"
        context="YOLOv8 prohibited-item detection"
    """

    name: str
    value: str
    context: str | None = None


class ExtractedFact(BaseModel):
    """
    This is what Gemma is allowed to generate.

    Notice that source information is NOT here.
    Python adds provenance later.
    """

    claim: str = Field(
        min_length=3,
        description="Atomic factual claim supported by the source chunk."
    )

    category: EvidenceCategory

    skills: list[str] = Field(default_factory=list)

    metrics: list[ExtractedMetric] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    evidence: list[ExtractedFact] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Final evidence-card schema
# --------------------------------------------------------------------------- #

class EvidenceCard(BaseModel):
    evidence_id: str

    claim: str
    category: EvidenceCategory

    skills: list[str]
    metrics: list[ExtractedMetric]

    # Provenance
    source_chunk_id: str
    document_id: str
    source_file: str
    source_type: str
    page: int | None = None


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """
You are the evidence extraction component of ResumeForge.

Your job is to extract career-relevant factual evidence from a source chunk.

STRICT RULES:

1. Extract ONLY information explicitly supported by the source text.
2. Never invent technologies, numbers, responsibilities, outcomes, job titles,
   dates, skills, or accomplishments.
3. Do not improve or rewrite achievements to sound more impressive.
4. Do not make assumptions based on what would normally be true.
5. Each evidence item must contain ONE atomic claim.
6. Preserve numerical values exactly as they appear in the source.
7. Skills must be supported by the source text.
8. If the source contains no useful career evidence, return an empty list.
9. Do not extract vague filler statements.
10. Return valid JSON only.

The required output format is:

{
  "evidence": [
    {
      "claim": "A factual claim",
      "category": "project",
      "skills": ["Python", "FastAPI"],
      "metrics": [
        {
          "name": "latency",
          "value": "2-3 seconds",
          "context": "inference"
        }
      ]
    }
  ]
}

Allowed category values:

skill
project
experience
achievement
education
certification
responsibility
other
"""


def build_user_prompt(chunk: dict) -> str:
    return f"""
Extract resume-relevant evidence from the following source chunk.

SOURCE TYPE:
{chunk["source_type"]}

SOURCE FILE:
{chunk["source_file"]}

SOURCE TEXT:
--- BEGIN SOURCE ---
{chunk["text"]}
--- END SOURCE ---

Return JSON only.
""".strip()


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def load_state() -> dict[str, str]:
    """
    Maps:

        chunk_id -> SHA256 hash of chunk text

    This lets us avoid processing unchanged chunks again.
    """

    if not STATE_FILE.exists():
        return {}

    with STATE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_state(state: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False,
        )


def hash_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def make_evidence_id(
    chunk_id: str,
    claim: str,
) -> str:
    """
    Deterministic evidence IDs.

    If the same chunk produces the same claim,
    we get the same ID.
    """

    raw = f"{chunk_id}:{claim.strip().lower()}"

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return f"ev_{digest[:16]}"


def deduplicate_strings(values: list[str]) -> list[str]:
    """
    Deduplicate while preserving original order.
    """

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


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #

def extract_json_object(raw: str) -> dict:
    """
    Gemma should return JSON only.

    This fallback handles cases where it still returns:

        ```json
        {...}
        ```

    or adds a small amount of surrounding text.
    """

    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output did not contain a JSON object.")

    candidate = text[start:end + 1]

    return json.loads(candidate)


# --------------------------------------------------------------------------- #
# Model interaction
# --------------------------------------------------------------------------- #

def extract_facts(
    llm: LLM,
    chunk: dict,
    retries: int = 2,
) -> list[ExtractedFact]:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(chunk),
        },
    ]

    config = GenConfig(
        max_new_tokens=1000,
        temperature=0.0,
        top_p=1.0,
    )

    last_error: Exception | None = None

    for attempt in range(retries + 1):

        raw_response = llm.generate(
            messages,
            config=config,
        )

        try:
            parsed_json = extract_json_object(
                raw_response
            )

            response = ExtractionResponse.model_validate(
                parsed_json
            )

            return response.evidence

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:

            last_error = error

            print(
                f"  Invalid model output "
                f"(attempt {attempt + 1}/{retries + 1})"
            )

            # Give the model its bad output and ask it
            # to repair the structure.
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": raw_response,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid. "
                            "Return the same extraction as valid JSON "
                            "matching the required schema. "
                            "Return JSON only."
                        ),
                    },
                ]
            )

    raise RuntimeError(
        f"Could not extract valid evidence: {last_error}"
    )


# --------------------------------------------------------------------------- #
# Evidence-card construction
# --------------------------------------------------------------------------- #

def build_evidence_cards(
    chunk: dict,
    facts: list[ExtractedFact],
) -> list[EvidenceCard]:

    cards = []

    for fact in facts:

        claim = fact.claim.strip()

        skills = deduplicate_strings(
            fact.skills
        )

        card = EvidenceCard(
            evidence_id=make_evidence_id(
                chunk["chunk_id"],
                claim,
            ),

            claim=claim,
            category=fact.category,

            skills=skills,
            metrics=fact.metrics,

            source_chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            source_file=chunk["source_file"],
            source_type=chunk["source_type"],
            page=chunk.get("page"),
        )

        cards.append(card)

    return cards


# --------------------------------------------------------------------------- #
# Corpus extraction
# --------------------------------------------------------------------------- #

def extract_corpus(
    llm: LLM,
    chunks_file: Path = CHUNKS_FILE,
    evidence_file: Path = EVIDENCE_FILE,
) -> None:

    chunks = load_jsonl(chunks_file)

    if not chunks:
        print("No chunks found.")
        return

    old_state = load_state()

    existing_records = load_jsonl(
        evidence_file
    )

    # Group existing evidence by source chunk.
    existing_by_chunk: dict[str, list[dict]] = {}

    for record in existing_records:

        chunk_id = record["source_chunk_id"]

        existing_by_chunk.setdefault(
            chunk_id,
            []
        ).append(record)

    new_state: dict[str, str] = {}

    final_records: list[dict] = []

    processed = 0
    skipped = 0
    failed = 0
    total_evidence = 0

    print()
    print("ResumeForge Evidence Extractor")
    print("------------------------------")
    print(f"Chunks discovered: {len(chunks)}")
    print()

    for number, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk_id = chunk["chunk_id"]

        current_hash = hash_text(
            chunk["text"]
        )

        new_state[chunk_id] = current_hash

        # ------------------------------------------------------------ #
        # Skip unchanged chunks
        # ------------------------------------------------------------ #

        if (
            old_state.get(chunk_id)
            == current_hash
        ):

            old_cards = existing_by_chunk.get(
                chunk_id,
                []
            )

            final_records.extend(
                old_cards
            )

            total_evidence += len(
                old_cards
            )

            skipped += 1

            print(
                f"[{number}/{len(chunks)}] "
                f"SKIP {chunk_id}"
            )

            continue

        # ------------------------------------------------------------ #
        # Extract changed/new chunks
        # ------------------------------------------------------------ #

        print(
            f"[{number}/{len(chunks)}] "
            f"EXTRACT {chunk_id}"
        )

        try:

            facts = extract_facts(
                llm,
                chunk,
            )

            cards = build_evidence_cards(
                chunk,
                facts,
            )

            final_records.extend(
                card.model_dump()
                for card in cards
            )

            total_evidence += len(cards)
            processed += 1

            print(
                f"  -> {len(cards)} evidence cards"
            )

        except Exception as error:

            failed += 1

            print(
                f"  FAILED: {error}"
            )

            # Important:
            # don't mark failed chunks as successfully processed.
            new_state.pop(
                chunk_id,
                None,
            )

            # Preserve old evidence if it existed.
            old_cards = existing_by_chunk.get(
                chunk_id,
                []
            )

            final_records.extend(
                old_cards
            )

    # ---------------------------------------------------------------- #
    # Save
    # ---------------------------------------------------------------- #

    save_jsonl(
        evidence_file,
        final_records,
    )

    save_state(
        new_state
    )

    print()
    print("Extraction complete")
    print("-------------------")
    print(f"New/changed processed: {processed}")
    print(f"Unchanged skipped:     {skipped}")
    print(f"Failed:                {failed}")
    print(f"Evidence cards:        {total_evidence}")
    print(f"Output:                {evidence_file}")