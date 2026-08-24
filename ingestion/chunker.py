from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


INPUT_FILE = Path("../corpus/processed/documents.jsonl")
OUTPUT_FILE = Path("../corpus/processed/chunks.jsonl")

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200
OVERLAP_CHARS = 150


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source_file: str
    source_type: str
    text: str
    page: int | None
    chunk_index: int


def load_documents(
    input_file: Path = INPUT_FILE,
) -> list[dict]:
    """
    Load parsed documents from JSONL.
    """
    documents = []

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            documents.append(json.loads(line))

    return documents


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text using paragraph boundaries.

    Multiple newlines are treated as paragraph separators.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def split_large_paragraph(paragraph: str) -> list[str]:
    """
    Split an oversized paragraph into sentences.

    This is intentionally lightweight.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        paragraph,
    )

    pieces = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate

        else:
            if current:
                pieces.append(current)

            current = sentence

    if current:
        pieces.append(current)

    return pieces


def normalize_paragraphs(text: str) -> list[str]:
    """
    Return paragraphs small enough to participate
    in normal chunk construction.
    """

    paragraphs = split_into_paragraphs(text)

    normalized = []

    for paragraph in paragraphs:

        if len(paragraph) <= MAX_CHUNK_CHARS:
            normalized.append(paragraph)

        else:
            normalized.extend(
                split_large_paragraph(paragraph)
            )

    return normalized


def get_overlap(text: str) -> str:
    """
    Preserve a small amount of text from the previous
    chunk so context is not completely lost.
    """

    if len(text) <= OVERLAP_CHARS:
        return text

    overlap = text[-OVERLAP_CHARS:]

    # Avoid beginning the overlap halfway through a word.
    first_space = overlap.find(" ")

    if first_space != -1:
        overlap = overlap[first_space + 1:]

    return overlap.strip()


def chunk_text(text: str) -> list[str]:
    """
    Create chunks while trying to preserve paragraph
    boundaries.
    """

    paragraphs = normalize_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        candidate = (
            f"{current_chunk}\n\n{paragraph}".strip()
            if current_chunk
            else paragraph
        )

        if len(candidate) <= MAX_CHUNK_CHARS:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

            overlap = get_overlap(current_chunk)

            current_chunk = (
                f"{overlap}\n\n{paragraph}".strip()
                if overlap
                else paragraph
            )

        else:
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return merge_tiny_chunks(chunks)


def merge_tiny_chunks(chunks: list[str]) -> list[str]:
    """
    Avoid producing tiny fragments where possible.
    """

    if len(chunks) <= 1:
        return chunks

    merged = []

    for chunk in chunks:

        if (
            merged
            and len(chunk) < MIN_CHUNK_CHARS
            and len(merged[-1]) + len(chunk) <= MAX_CHUNK_CHARS
        ):
            merged[-1] = (
                f"{merged[-1]}\n\n{chunk}"
            )

        else:
            merged.append(chunk)

    return merged


def create_chunks(
    documents: list[dict],
) -> list[Chunk]:
    """
    Convert normalized documents into retrievable chunks.
    """

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for index, text in enumerate(document_chunks):

            chunk_id = (
                f"{document['document_id']}"
                f"_chunk_{index:03d}"
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document["document_id"],
                    source_file=document["source_file"],
                    source_type=document["source_type"],
                    text=text,
                    page=document.get("page"),
                    chunk_index=index,
                )
            )

    return chunks


def save_chunks(
    chunks: list[Chunk],
    output_file: Path = OUTPUT_FILE,
) -> None:
    """
    Save chunks to JSONL.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:

            json.dump(
                asdict(chunk),
                file,
                ensure_ascii=False,
            )

            file.write("\n")


def main() -> None:

    documents = load_documents()

    chunks = create_chunks(documents)

    save_chunks(chunks)

    print()
    print("ResumeForge Chunker")
    print("-------------------")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created:   {len(chunks)}")
    print(f"Output:           {OUTPUT_FILE}")


if __name__ == "__main__":
    main()