from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from config import CHUNKS_FILE, EVIDENCE_FILE, STATE_FILE
from local_llm import GenConfig, LLM


# --------------------------------------------------------------------------- #
# Cache / extraction configuration
# --------------------------------------------------------------------------- #

STATE_VERSION = 2

EXTRACTION_MAX_NEW_TOKENS = 512
EXTRACTION_TEMPERATURE = 0.0
EXTRACTION_TOP_P = 1.0
EXTRACTION_RETRIES = 2


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
    """

    name: str
    value: str
    context: str | None = None


class ExtractedFact(BaseModel):
    """
    Information the LLM is allowed to extract.

    Source/provenance information is intentionally not generated
    by the model. Python attaches it later from the source chunk.
    """

    claim: str = Field(
        min_length=3,
        description="Atomic factual claim supported by the source chunk.",
    )

    category: EvidenceCategory

    skills: list[str] = Field(
        default_factory=list,
    )

    metrics: list[ExtractedMetric] = Field(
        default_factory=list,
    )


class ExtractionResponse(BaseModel):
    evidence: list[ExtractedFact] = Field(
        default_factory=list,
    )


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
11. Extract at most 8 high-value evidence items from each chunk.
12. Prefer concrete resume-relevant evidence over trivial biographical facts.
13. Do not infer current status. For example, a completed degree does not imply
    the person is currently a student.
14. The JSON example below describes STRUCTURE ONLY. Never copy claims,
    technologies, skills, metric names, metric values, or context from it.
15. A metric may be included ONLY when its exact value is explicitly present
    in the source text.
16. Do not convert a listed skill into "proficient", "expert", "experienced",
    or similar stronger wording unless the source explicitly supports that
    wording.

The required output format is:

{
  "evidence": [
    {
      "claim": "<atomic factual claim from source>",
      "category": "project",
      "skills": ["<skill explicitly present in source>"],
      "metrics": [
        {
          "name": "<metric name from source>",
          "value": "<exact metric value from source>",
          "context": "<optional context from source>"
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
    """
    Build the source-specific extraction request.
    """

    return f"""
Extract resume-relevant evidence from the following source chunk.

SOURCE TYPE:
{chunk["source_type"]}

SOURCE FILE:
{chunk["source_file"]}

PAGE:
{chunk.get("page")}

SOURCE TEXT:
--- BEGIN SOURCE ---
{chunk["text"]}
--- END SOURCE ---

Return JSON only.
""".strip()


# --------------------------------------------------------------------------- #
# JSONL utilities
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    """
    Load a JSONL file.

    Missing files are treated as empty.
    """

    if not path.exists():
        return []

    records: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def save_jsonl_atomic(
    path: Path,
    records: list[dict],
) -> None:
    """
    Atomically replace a JSONL file.

    The complete output is first written to a temporary file in the
    destination directory. os.replace() then swaps it into place.

    This prevents a crash during writing from leaving a partially
    truncated evidence file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )

    temp_path = Path(
        temp_name
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:
            for record in records:
                json.dump(
                    record,
                    file,
                    ensure_ascii=False,
                )

                file.write("\n")

            file.flush()
            os.fsync(
                file.fileno()
            )

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        temp_path.unlink(
            missing_ok=True,
        )

        raise


# --------------------------------------------------------------------------- #
# State / cache utilities
# --------------------------------------------------------------------------- #

def empty_state() -> dict:
    """
    Return a fresh cache-state structure.
    """

    return {
        "version": STATE_VERSION,
        "chunks": {},
    }


def load_state(
    path: Path,
) -> dict:
    """
    Load evidence extraction state.

    Expected structure:

    {
        "version": 2,
        "chunks": {
            "chunk_id": {
                "fingerprint": "...",
                "evidence_ids": [...]
            }
        }
    }

    Old or malformed state formats are invalidated rather than trusted.
    """

    if not path.exists():
        return empty_state()

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return empty_state()

    if not isinstance(
        state,
        dict,
    ):
        return empty_state()

    if state.get(
        "version"
    ) != STATE_VERSION:
        return empty_state()

    if not isinstance(
        state.get("chunks"),
        dict,
    ):
        return empty_state()

    return state


def save_state_atomic(
    path: Path,
    state: dict,
) -> None:
    """
    Atomically replace the state file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )

    temp_path = Path(
        temp_name
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                state,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.flush()
            os.fsync(
                file.fileno()
            )

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        temp_path.unlink(
            missing_ok=True,
        )

        raise


# --------------------------------------------------------------------------- #
# Cache fingerprinting
# --------------------------------------------------------------------------- #

def get_model_revision(
    llm: LLM,
) -> str | None:
    """
    Best-effort lookup of the resolved Hugging Face model revision.

    Transformers models commonly expose the downloaded repository
    commit through model.config._commit_hash.

    Backends that do not expose this simply return None.
    """

    model = getattr(
        llm,
        "model",
        None,
    )

    config = getattr(
        model,
        "config",
        None,
    )

    return getattr(
        config,
        "_commit_hash",
        None,
    )


def make_extraction_hash(
    chunk: dict,
    llm: LLM,
) -> str:
    """
    Fingerprint everything that materially affects evidence extraction.

    Re-extraction occurs when any relevant input changes, including:

    - source text
    - source provenance
    - model name
    - model backend
    - resolved model revision
    - system prompt
    - generated user prompt
    - generation settings
    - response schema
    """

    payload = {
        # Source / provenance
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_file": chunk["source_file"],
        "source_type": chunk["source_type"],
        "page": chunk.get("page"),
        "text": chunk["text"],

        # LLM identity
        "model_name": llm.name,
        "backend": type(llm).__name__,
        "model_revision": get_model_revision(
            llm
        ),

        # Actual prompts
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(
            chunk
        ),

        # Generation behavior
        "generation": {
            "max_new_tokens": EXTRACTION_MAX_NEW_TOKENS,
            "temperature": EXTRACTION_TEMPERATURE,
            "top_p": EXTRACTION_TOP_P,
            "retries": EXTRACTION_RETRIES,
        },

        # Validation contract
        "schema": ExtractionResponse.model_json_schema(),
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Evidence IDs / cleanup
# --------------------------------------------------------------------------- #

def make_evidence_id(
    chunk_id: str,
    claim: str,
) -> str:
    """
    Produce a deterministic evidence ID.

    The same claim from the same chunk receives the same ID.
    """

    raw = (
        f"{chunk_id}:"
        f"{claim.strip().lower()}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return f"ev_{digest[:16]}"


def deduplicate_strings(
    values: list[str],
) -> list[str]:
    """
    Deduplicate strings case-insensitively while preserving order.
    """

    seen: set[str] = set()
    result: list[str] = []

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

def extract_json_object(
    raw: str,
) -> dict:
    """
    Parse a JSON object from an LLM response.

    Handles:

    - plain JSON
    - ```json fenced responses
    - small amounts of surrounding text
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
            "Model output did not contain a JSON object."
        )

    candidate = text[
        start:end + 1
    ]

    return json.loads(
        candidate
    )


# --------------------------------------------------------------------------- #
# Model interaction
# --------------------------------------------------------------------------- #

def extract_facts(
    llm: LLM,
    chunk: dict,
    retries: int = EXTRACTION_RETRIES,
) -> list[ExtractedFact]:
    """
    Ask the LLM to extract structured evidence from one source chunk.

    Invalid JSON/schema responses are retried with a repair instruction.
    """

    messages: list[dict] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(
                chunk
            ),
        },
    ]

    config = GenConfig(
        max_new_tokens=EXTRACTION_MAX_NEW_TOKENS,
        temperature=EXTRACTION_TEMPERATURE,
        top_p=EXTRACTION_TOP_P,
    )

    last_error: Exception | None = None

    for attempt in range(
        retries + 1
    ):
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
                f"(attempt {attempt + 1}/{retries + 1}): "
                f"{error}"
            )

            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": raw_response,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON response is syntactically invalid.\n\n"
                            f"JSON parser error:\n{error}\n\n"
                            "Repair ONLY the JSON syntax.\n"
                            "Do not add, remove, rewrite, or reinterpret any factual claims.\n"
                            "Preserve the same evidence items, skills, metrics, and values.\n"
                            "Ensure all commas, brackets, braces, strings, and arrays are valid.\n"
                            "Return the complete corrected JSON object only."
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
    """
    Convert model-generated facts into provenance-backed EvidenceCards.
    """

    cards: list[EvidenceCard] = []

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

            source_chunk_id=chunk[
                "chunk_id"
            ],
            document_id=chunk[
                "document_id"
            ],
            source_file=chunk[
                "source_file"
            ],
            source_type=chunk[
                "source_type"
            ],
            page=chunk.get(
                "page"
            ),
        )

        cards.append(
            card
        )

    return cards


# --------------------------------------------------------------------------- #
# Cache coherence
# --------------------------------------------------------------------------- #

def evidence_ids_for_records(
    records: list[dict],
) -> list[str]:
    """
    Return sorted evidence IDs for cache-coherence checks.
    """

    return sorted(
        record["evidence_id"]
        for record in records
        if "evidence_id" in record
    )


def cache_entry_is_valid(
    *,
    evidence_cache_exists: bool,
    old_chunk_state: dict | None,
    current_fingerprint: str,
    old_cards: list[dict],
) -> bool:
    """
    Decide whether an existing chunk can safely be skipped.

    A fingerprint match alone is not sufficient.

    The corresponding evidence file must exist and contain exactly the
    evidence IDs recorded in state.
    """

    if not evidence_cache_exists:
        return False

    if not old_chunk_state:
        return False

    if old_chunk_state.get(
        "fingerprint"
    ) != current_fingerprint:
        return False

    expected_ids = sorted(
        old_chunk_state.get(
            "evidence_ids",
            [],
        )
    )

    actual_ids = evidence_ids_for_records(
        old_cards
    )

    return expected_ids == actual_ids


# --------------------------------------------------------------------------- #
# Corpus extraction
# --------------------------------------------------------------------------- #

def extract_corpus(
    llm: LLM,
    chunks_file: Path = CHUNKS_FILE,
    evidence_file: Path = EVIDENCE_FILE,
    state_file: Path = STATE_FILE,
) -> None:
    """
    Extract evidence for every corpus chunk.

    The cache is fail-closed:

    - missing chunks abort extraction
    - changed/new chunks must extract successfully
    - stale evidence is never silently reused after a failed extraction
    - orphaned state cannot suppress extraction
    - evidence is written before state
    - both files are replaced atomically
    """

    chunks = load_jsonl(
        chunks_file
    )

    if not chunks:
        raise RuntimeError(
            f"No chunks found in {chunks_file}. "
            "Evidence extraction cannot continue."
        )

    old_state = load_state(
        state_file
    )

    evidence_cache_exists = (
        evidence_file.exists()
    )

    existing_records = load_jsonl(
        evidence_file
    )

    # Group existing evidence by source chunk.
    existing_by_chunk: dict[
        str,
        list[dict],
    ] = {}

    for record in existing_records:
        chunk_id = record.get(
            "source_chunk_id"
        )

        if not chunk_id:
            continue

        existing_by_chunk.setdefault(
            chunk_id,
            [],
        ).append(
            record
        )

    new_state = empty_state()

    final_records: list[dict] = []

    processed = 0
    skipped = 0
    failed = 0
    total_evidence = 0

    print()
    print(
        "ResumeForge Evidence Extractor"
    )
    print(
        "------------------------------"
    )
    print(
        f"Chunks discovered: {len(chunks)}"
    )
    print()

    for number, chunk in enumerate(
        chunks,
        start=1,
    ):
        chunk_id = chunk[
            "chunk_id"
        ]

        current_fingerprint = make_extraction_hash(
            chunk,
            llm,
        )

        old_chunk_state = (
            old_state["chunks"].get(
                chunk_id
            )
        )

        old_cards = existing_by_chunk.get(
            chunk_id,
            [],
        )

        # ------------------------------------------------------------ #
        # Skip only when BOTH state and evidence are coherent.
        # ------------------------------------------------------------ #

        if cache_entry_is_valid(
            evidence_cache_exists=evidence_cache_exists,
            old_chunk_state=old_chunk_state,
            current_fingerprint=current_fingerprint,
            old_cards=old_cards,
        ):
            final_records.extend(
                old_cards
            )

            evidence_ids = evidence_ids_for_records(
                old_cards
            )

            new_state["chunks"][
                chunk_id
            ] = {
                "fingerprint": current_fingerprint,
                "evidence_ids": evidence_ids,
            }

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
        # Extract changed/new/incoherent chunks.
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

            card_records = [
                card.model_dump()
                for card in cards
            ]

            final_records.extend(
                card_records
            )

            evidence_ids = sorted(
                card.evidence_id
                for card in cards
            )

            # State is recorded only AFTER extraction succeeds.
            new_state["chunks"][
                chunk_id
            ] = {
                "fingerprint": current_fingerprint,
                "evidence_ids": evidence_ids,
            }

            total_evidence += len(
                cards
            )

            processed += 1

            print(
                f"  -> {len(cards)} evidence cards"
            )

        except Exception as error:
            failed += 1

            print(
                f"  FAILED: {error}"
            )

            raise RuntimeError(
                "Evidence extraction failed for "
                f"chunk {chunk_id}. "
                "Aborting to prevent stale evidence "
                "from being treated as current."
            ) from error

    # ---------------------------------------------------------------- #
    # Commit cache
    # ---------------------------------------------------------------- #
    #
    # Evidence is replaced FIRST.
    # State is replaced SECOND.
    #
    # If the process fails between these writes, the next execution
    # sees an evidence/state mismatch and safely re-extracts rather
    # than incorrectly skipping work.
    # ---------------------------------------------------------------- #

    save_jsonl_atomic(
        evidence_file,
        final_records,
    )

    save_state_atomic(
        state_file,
        new_state,
    )

    print()
    print(
        "Extraction complete"
    )
    print(
        "-------------------"
    )
    print(
        f"New/changed processed: {processed}"
    )
    print(
        f"Unchanged skipped:     {skipped}"
    )
    print(
        f"Failed:                {failed}"
    )
    print(
        f"Evidence cards:        {total_evidence}"
    )
    print(
        f"Output:                {evidence_file}"
    )
    print(
        f"State:                 {state_file}"
    )


# --------------------------------------------------------------------------- #
# Standalone test entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from local_llm import load

    llm = load(
        "gemma-2b"
    )

    extract_corpus(
        llm
    )