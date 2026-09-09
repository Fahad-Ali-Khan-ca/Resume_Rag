# Resume RAG

An evidence-grounded resume generation pipeline that turns a candidate's existing career documents into structured evidence and uses that evidence to generate job-specific resume content with a local LLM.

## Aim

The goal of this project is to generate high-quality, job-tailored resumes without inventing experience.

Instead of asking an LLM to rewrite a resume directly, the system first:

1. Parses the user's source documents.
2. Splits them into retrievable chunks.
3. Extracts structured evidence from those chunks.
4. Uses the job description to identify relevant evidence.
5. Generates resume content grounded in that evidence.
6. Renders the final output into a resume format.

This project is designed to become the resume-modification component of the larger **Autonomous Job Applier** system.

## Core Idea

```text
Candidate Documents
        │
        ▼
     Parser
        │
        ▼
     Chunker
        │
        ▼
Evidence Extractor
        │
        ▼
Structured Evidence
        │
        ├──────────────┐
        │              │
        ▼              ▼
Job Description   Resume Context
        │              │
        └──────┬───────┘
               ▼
           Local LLM
               │
               ▼
      Grounded Resume Content
               │
               ▼
          LaTeX Renderer
               │
               ▼
            Resume
```

## Current Project Structure

```text
Resume_Rag/
│
├── ingestion/
│   ├── parser.py
│   ├── chunker.py
│   └── evidence_extractor.py
│
├── rendering/
│   └── latex_renderer.py
│
├── llm.py
├── local_llm.py
├── run_pipeline.py
├── pyproject.toml
├── uv.lock
│
├── corpus/
│   └── candidate source documents
│
├── jobs/
│   └── job descriptions
│
└── docs/
    ├── SETUP.md
    ├── RUN.md
    └── ARCHITECTURE.md
```

The exact structure may evolve as the remaining ranking, generation, validation, and rendering stages are completed.

## Supported Input Documents

The ingestion pipeline is intended to work with:

- PDF
- Markdown
- Plain text

These documents form the evidence corpus used by the pipeline.

## Requirements

- Python 3.12 recommended
- `uv`
- PyTorch
- Hugging Face Transformers
- A compatible local model
- NVIDIA GPU recommended for local inference

CPU execution is possible but significantly slower for LLM inference.

### GPU Notes

#### NVIDIA

The project has been tested with CUDA-enabled PyTorch.

Example hardware used during development:

```text
NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

#### AMD on Windows

Native PyTorch ROCm support is currently not available for standard Windows installations.

On an AMD Windows machine, the project may therefore fall back to CPU unless another supported inference backend is configured.

See [docs/SETUP.md](docs/SETUP.md) for installation details.

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Resume_Rag
```

### 2. Install `uv`

Follow the official installation instructions:

https://docs.astral.sh/uv/getting-started/installation/

### 3. Create the virtual environment

```bash
uv venv --python 3.12
```

Activate it.

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
uv sync
```

If PyTorch needs to be installed for a specific GPU configuration, follow the instructions in:

[docs/SETUP.md](docs/SETUP.md)

### 5. Add candidate documents

Place resume, project, work-history, or supporting documents in the corpus directory.

Example:

```text
corpus/
├── software_resume.pdf
├── project_notes.md
└── experience.txt
```

### 6. Add a job description

Example:

```text
jobs/test_job.txt
```

### 7. Run the pipeline

Example:

```bash
uv run python run_pipeline.py --model gemma-2b --jd jobs/test_job.txt --name "Candidate Name"
```

If the environment is already synchronized and you intentionally want to skip dependency synchronization:

```bash
uv run --no-sync python run_pipeline.py --model gemma-2b --jd jobs/test_job.txt --name "Candidate Name"
```

See [docs/RUN.md](docs/RUN.md) for more detail.

## Running Individual Components

The evidence extractor can also be executed independently:

```bash
uv run python -m ingestion.evidence_extractor
```

This is useful when debugging the ingestion pipeline before running the complete resume-generation workflow.

## Development Status

The project currently focuses on the evidence-grounded RAG pipeline.

Major areas include:

- document ingestion
- chunking
- evidence extraction
- local LLM abstraction
- resume generation
- LaTeX rendering
- evidence validation

Some components may still be under active development.

## Why Evidence Grounding Matters

A standard LLM resume generator can produce polished language while also introducing unsupported claims.

This system attempts to reduce that risk by making source evidence a first-class object in the generation pipeline.

The intended rule is:

```text
No resume claim should exist unless the system can trace it back to candidate evidence.
```

That makes the system better suited for autonomous resume generation, where there is no human manually reviewing every generated bullet.

## Documentation

- [Setup Guide](docs/SETUP.md)
- [Running the Pipeline](docs/RUN.md)
- [Architecture](docs/ARCHITECTURE.md)

## Larger Project

Resume RAG is intended to become one module inside the **Autonomous Job Applier**:

```text
Job Discovery
     │
     ▼
Job Scraper
     │
     ▼
Resume RAG
     │
     ▼
Application Package
     │
     ▼
Job Applier
```

The long-term goal is a system capable of discovering jobs, generating evidence-grounded application material, and submitting applications with minimal manual intervention.
