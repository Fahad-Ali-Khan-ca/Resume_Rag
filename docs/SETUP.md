# Setup Guide

This document explains how to create the development environment for Resume RAG.

## 1. Prerequisites

Install:

- Git
- Python 3.12
- `uv`

Check Python:

```bash
python --version
```

Check Git:

```bash
git --version
```

## 2. Clone the Repository

```bash
git clone <your-repository-url>
cd Resume_Rag
```

## 3. Install `uv`

`uv` is the Python package and environment manager used by this project.

Official installation guide:

https://docs.astral.sh/uv/getting-started/installation/

Verify:

```bash
uv --version
```

## 4. Create the Virtual Environment

```bash
uv venv --python 3.12
```

This creates:

```text
.venv/
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 5. Install Project Dependencies

If the repository contains a valid `uv.lock`:

```bash
uv sync
```

`uv` reads the project dependencies from:

```text
pyproject.toml
```

and installs the resolved versions from:

```text
uv.lock
```

## `pyproject.toml`

`pyproject.toml` is the main Python project configuration file.

It typically stores:

- project name
- Python version requirement
- runtime dependencies
- development dependencies
- package metadata
- tool configuration

Example conceptually:

```toml
[project]
name = "resume-rag"
requires-python = ">=3.10"

dependencies = [
    "transformers",
    "pypdf",
    "pydantic",
]
```

Do not manually edit the virtual environment to manage packages. Update the project dependency configuration instead.

## `uv.lock`

`uv.lock` records the exact dependency versions resolved by `uv`.

This improves reproducibility:

```text
pyproject.toml
      │
      ▼
dependency requirements
      │
      ▼
uv resolves versions
      │
      ▼
uv.lock
      │
      ▼
same environment across machines
```

The lock file should normally be committed to Git.

## Adding a Package

Example:

```bash
uv add pydantic
```

This updates both:

```text
pyproject.toml
uv.lock
```

## Removing a Package

```bash
uv remove pydantic
```

## Updating Dependencies

Update a package:

```bash
uv lock --upgrade-package transformers
uv sync
```

Upgrade all resolvable dependencies:

```bash
uv lock --upgrade
uv sync
```

Review dependency upgrades before committing them.

## PyTorch Installation

PyTorch installation depends on the machine.

Official selector:

https://pytorch.org/get-started/locally/

### NVIDIA GPU

Use a CUDA-enabled PyTorch build compatible with your system.

During development, the NVIDIA environment used a CUDA-enabled PyTorch installation.

Verify CUDA:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expected example:

```text
True
NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

### CPU

CPU inference works without CUDA.

Verify:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Output:

```text
False
```

This is not an error if CPU execution is intentional.

### AMD GPU on Windows

Standard PyTorch ROCm wheels are primarily targeted at Linux.

A Windows AMD GPU may therefore not be exposed through:

```python
torch.cuda.is_available()
```

For the current project, treat AMD Windows as CPU fallback unless a supported backend is explicitly configured.

Do not install Linux ROCm wheels into a Windows environment.

## Hugging Face Models

The project uses local Hugging Face models through `transformers`.

One model used during development is:

```text
google/gemma-2-2b-it
```

Some models require accepting a license or authenticating with Hugging Face.

Authenticate if required:

```bash
huggingface-cli login
```

or use the current Hugging Face CLI command supported by your installed package version.

Do not commit access tokens to Git.

## Recommended Directory Layout

```text
Resume_Rag/
│
├── corpus/
├── jobs/
├── ingestion/
├── rendering/
├── docs/
├── run_pipeline.py
├── llm.py
├── local_llm.py
├── pyproject.toml
└── uv.lock
```

## Verify the Environment

Run:

```bash
uv run python -c "import torch, transformers, pydantic, pypdf; print('Environment OK')"
```

Then verify the ingestion module:

```bash
uv run python -m ingestion.evidence_extractor
```

If both commands succeed, the environment is ready for pipeline execution.
