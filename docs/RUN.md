# Running Resume RAG

This document explains how to run the pipeline and individual development stages.

## Before Running

Confirm that dependencies are installed:

```bash
uv sync
```

Confirm that the project environment works:

```bash
uv run python -c "import torch; print('Torch:', torch.__version__)"
```

## Prepare the Corpus

Place source documents inside the corpus directory.

Example:

```text
corpus/
├── resume.pdf
├── project_history.md
├── software_projects.pdf
└── experience.txt
```

The corpus should contain factual source material about the candidate.

The quality of this evidence directly affects the quality of the generated resume.

## Prepare the Job Description

Create a plain-text job description.

Example:

```text
jobs/software_engineer.txt
```

Paste the complete job posting into that file.

## Run the Full Pipeline

Example:

```bash
uv run python run_pipeline.py \
  --model gemma-2b \
  --jd jobs/software_engineer.txt \
  --name "Candidate Name"
```

### Windows PowerShell

The same command can be run on one line:

```powershell
uv run python run_pipeline.py --model gemma-2b --jd jobs/software_engineer.txt --name "Candidate Name"
```

## CLI Arguments

The current pipeline command follows this general form:

```text
run_pipeline.py
    --model <model alias>
    --jd <job description file>
    --name <candidate name>
```

### `--model`

Selects the configured local LLM.

Example:

```bash
--model gemma-2b
```

### `--jd`

Path to the target job description.

Example:

```bash
--jd jobs/test_job.txt
```

### `--name`

Candidate name inserted into the generated resume context.

Example:

```bash
--name "Candidate Name"
```

## Run Without Dependency Synchronization

Normally:

```bash
uv run ...
```

may ensure the project environment is synchronized.

When debugging an already-configured environment, you can intentionally skip synchronization:

```bash
uv run --no-sync python run_pipeline.py --model gemma-2b --jd jobs/test_job.txt --name "Candidate Name"
```

Use this only when you know the current virtual environment already contains the required dependencies.

## Run the Evidence Extractor

The evidence extraction stage can be tested independently:

```bash
uv run python -m ingestion.evidence_extractor
```

This is useful for confirming:

- documents are discovered
- parser output is valid
- chunks are generated
- the local model loads
- structured evidence is extracted

A successful development run previously discovered output similar to:

```text
Chunks discovered: 24
```

The exact number depends on the corpus.

## Typical Pipeline Flow

A complete execution is intended to follow this flow:

```text
1. Load local model

2. Parse candidate corpus

3. Split documents into chunks

4. Extract evidence from chunks

5. Read target job description

6. Identify relevant evidence

7. Generate resume sections

8. Validate generated claims

9. Render final resume
```

The exact stage numbering may change while the project is under development.

## Debugging by Stage

When the complete pipeline fails, test the earliest stage independently.

### Parser

Check that files are readable and supported.

Expected formats:

```text
.pdf
.md
.txt
```

### Chunker

Verify that parsed documents produce non-empty chunks.

### Evidence Extractor

Run:

```bash
uv run python -m ingestion.evidence_extractor
```

### Local LLM

Verify GPU detection:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Renderer

If LaTeX generation fails, test the renderer separately and verify that all imported resume data structures still match the renderer interface.

## Common Problems

### `nvcc` is not recognized

`nvcc` belongs to the CUDA Toolkit.

Its absence does not necessarily mean PyTorch cannot use the GPU.

The more important test is:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If this returns:

```text
True
```

then PyTorch can access CUDA.

### `torch.cuda.is_available()` returns `False`

Possible causes include:

- CPU-only PyTorch installation
- incompatible GPU driver
- AMD GPU on Windows
- unsupported PyTorch build
- incorrect environment

### Module import errors

Run commands from the repository root.

Correct:

```text
Resume_Rag/
> uv run python -m ingestion.evidence_extractor
```

Avoid running package modules from inside their subdirectory unless the code explicitly supports it.

## Recommended Development Workflow

Use the smallest failing stage first:

```text
Parser
  ↓
Chunker
  ↓
Evidence Extractor
  ↓
Retrieval
  ↓
Generation
  ↓
Validation
  ↓
Rendering
```

Only run the complete pipeline after the earlier stages succeed independently.
