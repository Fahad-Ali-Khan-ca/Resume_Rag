from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


RAW_DIR = Path("../corpus/raw") # Corpus is limited to PDF, Markdown, and TXT files.
OUTPUT_FILE = Path("../corpus/processed/documents.jsonl")


@dataclass
class ParsedDocument:
    document_id: str
    source_file: str
    source_type: str
    text: str
    page: int | None = None


def parse_pdf(file_path: Path) -> list[ParsedDocument]:
    """
    Parse a PDF into one document record per page.
    """
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text or not text.strip():
            continue

        documents.append(
            ParsedDocument(
                document_id=f"{file_path.stem}_page_{page_number}",
                source_file=str(file_path),
                source_type="pdf",
                text=clean_text(text),
                page=page_number,
            )
        )

    return documents


def parse_text_file(file_path: Path, source_type: str) -> list[ParsedDocument]:
    """
    Parse Markdown or TXT files.
    """
    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [
        ParsedDocument(
            document_id=file_path.stem,
            source_file=str(file_path),
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

def parse_file(file_path: Path) -> list[ParsedDocument]:
    """
    Choose the correct parser based on file extension.
    """
    extension = file_path.suffix.lower()

    match extension:
        case ".pdf":
            return parse_pdf(file_path)

        case ".md":
            return parse_text_file(file_path, "markdown")

        case ".txt":
            return parse_text_file(file_path, "txt")

        case _:
            print(f"Skipping unsupported file: {file_path}")
            return []


def parse_corpus(raw_dir: Path = RAW_DIR) -> list[ParsedDocument]:
    """
    Discover and parse every supported file inside corpus/raw.
    """
    documents = []

    supported_extensions = {".pdf", ".md", ".txt"}

    for file_path in raw_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        print(f"Parsing: {file_path}")

        try:
            parsed = parse_file(file_path)
            documents.extend(parsed)

        except Exception as error:
            print(f"Failed to parse {file_path}: {error}")

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
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        for document in documents:
            json.dump(
                asdict(document),
                file,
                ensure_ascii=False
            )

            file.write("\n")


def main() -> None:
    documents = parse_corpus()

    save_documents(documents)

    print()
    print(f"Parsed {len(documents)} document records.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()