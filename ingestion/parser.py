from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from config import RAW_DIR, DOCUMENTS_FILE
from ingestion.canonical_parser import (
    is_canonical_markdown,
    parse_canonical_markdown,
)


OUTPUT_FILE = DOCUMENTS_FILE


@dataclass
class ParsedDocument:
    document_id: str
    source_file: str
    source_type: str
    text: str
    page: int | None = None

    # Semantic ownership. Canonical Markdown sources populate these fields.
    entity_id: str | None = None
    entity_type: str | None = None

    # Deterministically parsed structure used by candidate_profile.py.
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


def parse_pdf(
    file_path: Path,
) -> list[ParsedDocument]:
    """
    Parse a PDF into one document record per page.

    PDFs are unstructured fallback sources, so entity ownership is unknown at
    this stage.
    """
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text()

        if not text or not text.strip():
            continue

        documents.append(
            ParsedDocument(
                document_id=(
                    f"{file_path.stem}"
                    f"_page_{page_number}"
                ),
                source_file=str(
                    file_path
                ),
                source_type="pdf",
                text=clean_text(text),
                page=page_number,
            )
        )

    return documents


def parse_canonical_file(
    file_path: Path,
) -> list[ParsedDocument]:
    """
    Parse profile.md / experience.md / projects.md into semantic entities.

    One physical Markdown file can produce many ParsedDocument records. This
    is intentional: entity boundaries become document boundaries before
    chunking, so evidence can never cross from one job/project into another.
    """
    entities = parse_canonical_markdown(
        file_path
    )

    return [
        ParsedDocument(
            document_id=(
                entity.document_id
            ),
            source_file=str(
                file_path
            ),
            source_type="markdown",
            text=clean_text(
                entity.text
            ),
            page=None,
            entity_id=entity.entity_id,
            entity_type=(
                entity.entity_type
            ),
            metadata=(
                entity.metadata
            ),
        )
        for entity in entities
        if entity.text.strip()
        or entity.metadata
    ]


def parse_text_file(
    file_path: Path,
    source_type: str,
) -> list[ParsedDocument]:
    """
    Parse arbitrary Markdown or TXT as an unstructured fallback document.
    """
    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if not text.strip():
        return []

    return [
        ParsedDocument(
            document_id=(
                file_path.stem
            ),
            source_file=str(
                file_path
            ),
            source_type=source_type,
            text=clean_text(text),
        )
    ]


def clean_text(text: str) -> str:
    lines = []
    previous_blank = False

    for line in text.splitlines():
        line = line.strip()

        if not line:
            if not previous_blank:
                lines.append("")

            previous_blank = True
            continue

        lines.append(line)
        previous_blank = False

    return "\n".join(lines).strip()


def parse_file(
    file_path: Path,
) -> list[ParsedDocument]:
    """
    Choose the correct parser based on file extension/source contract.
    """
    if is_canonical_markdown(
        file_path
    ):
        return parse_canonical_file(
            file_path
        )

    extension = file_path.suffix.lower()

    match extension:
        case ".pdf":
            return parse_pdf(
                file_path
            )

        case ".md":
            return parse_text_file(
                file_path,
                "markdown",
            )

        case ".txt":
            return parse_text_file(
                file_path,
                "txt",
            )

        case _:
            print(
                f"Skipping unsupported file: "
                f"{file_path}"
            )
            return []


def parse_corpus(
    raw_dir: Path = RAW_DIR,
) -> list[ParsedDocument]:
    """
    Discover and parse every supported file inside corpus/raw.
    """
    documents = []

    supported_extensions = {
        ".pdf",
        ".md",
        ".txt",
    }

    for file_path in sorted(
        raw_dir.rglob("*")
    ):
        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in supported_extensions
        ):
            continue

        print(
            f"Parsing: {file_path}"
        )

        try:
            parsed = parse_file(
                file_path
            )
            documents.extend(parsed)

        except Exception as error:
            # Canonical files are source-of-truth inputs. Silently skipping a
            # malformed one would create an incomplete resume, so fail closed.
            if is_canonical_markdown(
                file_path
            ):
                raise RuntimeError(
                    "Failed to parse canonical corpus file "
                    f"{file_path}: {error}"
                ) from error

            print(
                f"Failed to parse {file_path}: "
                f"{error}"
            )

    return documents


def save_documents(
    documents: list[ParsedDocument],
    output_file: Path = OUTPUT_FILE,
) -> None:
    """
    Save normalized documents as JSONL.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            json.dump(
                asdict(document),
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def main() -> None:
    documents = parse_corpus()
    save_documents(documents)

    print()
    print(
        f"Parsed {len(documents)} "
        "document records."
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
