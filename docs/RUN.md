# Running ResumeForge v1

Run commands from the repository root. The full pipeline writes its outputs to `output/`.

## 1. Prepare candidate data

Create private files under `corpus/raw/`. The preferred input is **three canonical Markdown documents** rather than an arbitrary resume PDF: this provides explicit employer and project boundaries. The example below is structural only; replace every value with the candidate's own verifiable information.

`corpus/raw/profile.md`:

```markdown
# Candidate
## Contact
Name: Candidate Name
Email: candidate@example.com
Location: Toronto, ON
LinkedIn: linkedin.com/in/example
GitHub: github.com/example

## Skills
- Python
- SQL
- Git

## Education
### Example University
Degree: Bachelor of Engineering
Field: Software Engineering
Location: Toronto, ON
Graduation: August 2025
```

`corpus/raw/experience.md`:

```markdown
## exp_example_employer_2024
Company: Example Employer
Title: Software Engineer Intern
Location: Toronto, ON
Start: 2024-05
End: 2024-08

### Responsibilities
- Implemented a documented internal API.

### Achievements
- Increased automated test coverage from a verified source measurement.
```

`corpus/raw/projects.md`:

```markdown
## project_example
Name: Example Project
Date: 2025

### Technologies
- Python
- Docker

### Features
- Built a documented data ingestion workflow.
```

Use one `##` entry per employer/project. Keep employer/project facts under their own entry rather than a shared section. Add only truthful, source-backed accomplishments and measurements. Additional PDFs, Markdown and TXT files are accepted as fallback evidence, but their attribution is less reliable than canonical entities. Do not commit personal corpus data.

## 2. Prepare the job posting

Save its full text in a file such as `jobs/test_job.txt`. Check whether the posting and extracted requirements actually match the position you intend to apply for.

## 3. Run the full pipeline

**Windows PowerShell, NVIDIA configuration used for v1 testing:**

```powershell
uv sync --extra nvidia
pdflatex --version
uv run --extra nvidia python run_pipeline.py --model gemma-4-e2b-q4 --jd jobs/test_job.txt --name "Candidate Name"
```

A single-line command avoids PowerShell continuation issues. The first model run downloads its model weights; repeated runs can reuse cached evidence when model and sources are unchanged.

Alternative configured model aliases include `gemma-2b-q4` (llama.cpp) and `gemma-4-e2b` (full Transformers checkpoint; substantially higher memory requirements). Not all aliases or GPU extras will work on every operating system or hardware configuration. See [Setup](SETUP.md).

## CLI parameters

`--jd` (required) takes a text job posting. `--model` selects a `local_llm.py` registry alias (default: `gemma-2b`). Optional overrides: `--name`, `--email`, `--phone`, `--location`, `--linkedin`, `--github`. The current full pipeline uses US Letter paper; A4 is supported by the renderer's Python API but not exposed as a CLI option.

## What success produces

```text
output/
  tailored_resume.json   # final fitted structured content and evidence IDs
  tailored_resume.tex    # editable LaTeX and evidence-ID comments
  tailored_resume.pdf    # compiled, verified one-page document
  layout_report.json     # density, page count, bullet counts, warnings
```

The console should end with `Pages: 1` and `ResumeForge complete.`. If `layout_report.json` warns that an entity has fewer validated bullets than its target, add **distinct, truthful source material** rather than forcing the model to generate filler. If `estimated_bottom_usage` is `null`, the optional PDF text-position measurement was unavailable; page count was still checked.

The `output/` files are replaced on each run. To compare two models:

```powershell
Copy-Item .\output .\output_gemma4_e2b_q4 -Recurse
```

## Debugging

| Symptom | What to check |
| --- | --- |
| `pdflatex is required` | Install a LaTeX distribution and confirm `pdflatex --version` works in the same terminal. |
| LaTeX compilation fails | Read the reported `pdflatex` error; install missing template packages and inspect `output/tailored_resume.tex` if it exists. |
| One-page fitting raises | Review the selected entries and required content; the fitter refuses to destroy core evidence just to fit. |
| Too few bullets | Inspect `layout_report.json`, canonical source quality, extracted evidence, entity ownership and claim-validation results. |
| Repeated bullets or bad skills | These are known v1 quality limitations; edit canonical sources as appropriate and review the final PDF manually. |
| Unreasonable STRONG experience match | Verify required years and actual hands-on work independently; match labels are not an eligibility guarantee. |
| Unexpected extraction work | The evidence cache changes with model identity or source changes. |
| Import error | Run from the repository root with the correct `uv` environment/extra. |

For developer diagnostics, `uv run --extra nvidia python -m ingestion.evidence_extractor` runs the extractor independently once its inputs are prepared. Read [Rendering](RENDERING.md) for page-fit behavior and [Architecture](ARCHITECTURE.md) for the full data flow.

**Before submission:** check that every skill, employer, date, project, metric and requirement match is accurate, and visually inspect the generated one-page PDF.
