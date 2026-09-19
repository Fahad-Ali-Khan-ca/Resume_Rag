# ResumeForge (Resume RAG)

ResumeForge is a **local, evidence-grounded resume tailoring pipeline**. It parses candidate career documents, extracts traceable evidence, matches that evidence against a job description, generates and validates resume bullets, and composes a **single-column, one-page LaTeX/PDF resume**.

ResumeForge is the resume-generation component of the larger Autonomous Job Applier project. **v1 generates resumes; it does not scrape jobs or submit applications.** Generated claims and job-requirement matches still require human review before applying.

## How v1 works

```text
Candidate corpus (profile.md, experience.md, projects.md; optional PDF/TXT/MD)
     |
Canonical entity parser + candidate profile (stable experience/project IDs)
     |
Entity-bounded chunks -> local LLM evidence extraction -> evidence cache
     |
Hybrid retrieval (BM25 + semantic) + job-requirement analysis/matching
     |
Evidence-bounded resume planner -> bullet generation -> claim validation
     |
Composition policy (entity coverage, verified bullet targets, skill selection)
     |
One-column LaTeX template -> PDF compilation -> page-count/spacing fit
     |
output/tailored_resume.json + .tex + .pdf + layout_report.json
```

The canonical parser keeps an employer's or project's source facts within that entity. Bullets must reference evidence owned by their entry; a metric belonging to one project must not be reassigned to another. The validator checks generated statements against the supplied evidence, but model-generated evidence and validation are **not a guarantee of factual accuracy**.

## Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Git.
- A compatible local model. The tested Windows/NVIDIA configuration uses `gemma-4-e2b-q4` with `llama.cpp`. Model files are downloaded on first use.
- A LaTeX distribution providing **`pdflatex` on PATH**, including the packages used by the template (`geometry`, `enumitem`, `hyperref`, `titlesec`, `lmodern`, `microtype`, and `fontenc`). The verified one-page PDF step requires `pdflatex`.
- GPU acceleration is optional in principle, but model size, installed wheels, platform and inference backend determine practical performance. See [Setup](docs/SETUP.md).

## Quick start — Windows PowerShell / NVIDIA

```powershell
git clone https://github.com/Fahad-Ali-Khan-ca/Resume_Rag.git
cd Resume_Rag
uv venv --python 3.12
uv sync --extra nvidia
pdflatex --version
```

Create **private** `corpus/raw/profile.md`, `corpus/raw/experience.md`, and `corpus/raw/projects.md` using the canonical structure described in [Run guide](docs/RUN.md); put a job description in `jobs/test_job.txt`.

```powershell
uv run --extra nvidia python run_pipeline.py --model gemma-4-e2b-q4 --jd jobs/test_job.txt --name "Candidate Name"
```

The `--name`, `--email`, `--phone`, `--location`, `--linkedin`, and `--github` options override contact fields from `profile.md`. See [Run guide](docs/RUN.md) for details and troubleshooting.

## Outputs

| File | Purpose |
| --- | --- |
| `output/tailored_resume.json` | Final **fitted** structured resume and per-bullet evidence IDs. |
| `output/tailored_resume.tex` | Editable one-column LaTeX document; evidence IDs also appear in comments. |
| `output/tailored_resume.pdf` | Compiled resume, checked for exactly one page. |
| `output/layout_report.json` | Page count, layout density, entry/bullet counts, optional estimated bottom-page utilization, and warnings. |

The output directory is overwritten by subsequent runs. Save comparison runs in another directory if needed. Source documents, generated resumes, local model files and credentials should not be committed to this public repository.

## v1 rendering policy

`generation/composition.py` attempts up to **three validated bullets per selected experience** (up to three experiences) and **two per selected project** (up to two projects). It can include an entry only if verified achievement evidence is available; these are **targets, not guaranteed counts**. Content must remain owned by the correct career entity. `rendering/latex_renderer.py` formats a single-column, ATS-oriented resume with an optional summary, categorized skills, experience, projects, education and certifications.

`rendering/page_fit.py` compiles the PDF, checks its page count, and tries bounded spacing presets. If it overflows, the fitter can remove optional extra bullets/summary/skills, but must not fabricate content or erase the sole validated bullet of an entry to force a page. If a readable single page is impossible, it raises an error instead of silently producing two pages. Underfilled pages receive a warning when measurable; empty space is **not** filled with invented achievements. The current command uses US Letter paper; the renderer also supports A4 programmatically.

**Limitations:** v1 may select repetitive bullets or overbroad skill labels, and requirement matching can overstate experience-duration or hands-on matches. Review all claims, skills, dates, quantitative requirements, and the final PDF before submitting it.

## Repository map

```text
ingestion/    canonical_parser.py, parser.py, chunker.py, evidence_extractor.py
retrieval/    lexical, semantic and hybrid evidence retrieval
jd/          job description requirement extraction
matching/    evidence/requirement matching
generation/  candidate_profile.py, planner.py, bullet_generator.py,
             resume_generator.py, composition.py
validation/  claim_validator.py
rendering/   latex_renderer.py, page_fit.py
run_pipeline.py
docs/        SETUP.md, RUN.md, ARCHITECTURE.md, RENDERING.md
```

## Documentation

- [Environment and LaTeX setup](docs/SETUP.md)
- [Canonical corpus, CLI, outputs and troubleshooting](docs/RUN.md)
- [End-to-end design and provenance](docs/ARCHITECTURE.md)
- [One-page composition and rendering contract](docs/RENDERING.md)
