from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CORPUS_DIR = PROJECT_ROOT / "corpus"
RAW_DIR = CORPUS_DIR / "raw"
PROCESSED_DIR = CORPUS_DIR / "processed"

DOCUMENTS_FILE = PROCESSED_DIR / "documents.jsonl"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EVIDENCE_FILE = PROCESSED_DIR / "evidence.jsonl"
STATE_FILE = PROCESSED_DIR / "evidence_state.json"