# ResumeForge v1 setup

This guide covers the Windows/NVIDIA configuration used to test the one-page rendering pipeline. Other hardware may require different model wheels or inference settings.

## Prerequisites

Install Git, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/) and a LaTeX distribution with `pdflatex` on PATH (MiKTeX or TeX Live). The template uses `geometry`, `enumitem`, `hyperref`, `titlesec`, `lmodern`, `microtype` and `fontenc`; make sure those packages are available.

```powershell
git --version
uv --version
pdflatex --version
```

The pipeline now **compiles and checks the PDF**, not just writes a `.tex` file. A working Python environment alone is insufficient.

## Clone and install — Windows PowerShell / NVIDIA

```powershell
git clone https://github.com/Fahad-Ali-Khan-ca/Resume_Rag.git
cd Resume_Rag
uv venv --python 3.12
uv sync --extra nvidia
```

You may activate the environment using `.\.venv\Scripts\Activate.ps1`, but `uv run` does not require manual activation. The NVIDIA extra selects the project's CUDA/PyTorch and configured `llama-cpp-python` wheel dependencies. GPU support depends on compatible drivers, wheel availability and model size.

Verify the installed environment:

```powershell
uv run --extra nvidia python -c "import pydantic, pypdf, llama_cpp; print('Python dependencies OK')"
pdflatex --version
```

For a PyTorch/CUDA backend check (distinct from the llama.cpp backend):

```powershell
uv run --extra nvidia python -c "import torch; print(torch.cuda.is_available())"
```

## Model choice

The tested v1 workflow uses `--model gemma-4-e2b-q4`, mapped in `local_llm.py` to `ggml-org/gemma-4-E2B-it-GGUF` / `gemma-4-E2B-it-Q4_0.gguf` via llama.cpp. The model is downloaded on first use. `gemma-2b-q4` is also registered. `gemma-4-e2b` is the full Transformers checkpoint and needs substantially more memory. Model availability, license acceptance, required access and GPU support depend on the selected checkpoint and local setup.

**AMD on Windows:** the repository's `amd` extra targets an ROCm-based PyTorch configuration; do not assume that it works on all Windows AMD systems. The NVIDIA-specific `llama-cpp-python` wheel is not a generic AMD acceleration package. Use a compatible local backend and verify it on the target machine.

## Private inputs and outputs

Create `corpus/raw/profile.md`, `corpus/raw/experience.md`, `corpus/raw/projects.md` with factual data, and a text job posting under `jobs/`. See [RUN.md](RUN.md) for canonical format. The repo ignores private raw/processed corpus and generated `output*/` files; check `git status` before pushing anything publicly.

## First complete run

```powershell
uv run --extra nvidia python run_pipeline.py --model gemma-4-e2b-q4 --jd jobs/test_job.txt --name "Candidate Name"
```

A successful run writes `output/tailored_resume.pdf`, `tailored_resume.tex`, `tailored_resume.json`, and `layout_report.json`. The PDF is required to compile to exactly one page; when it cannot fit without removing necessary content, the fitter raises instead of silently making an unreadable document. Consult [RENDERING.md](RENDERING.md) for page-fitting behavior.
