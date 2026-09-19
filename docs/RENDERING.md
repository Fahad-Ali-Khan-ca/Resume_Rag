# ResumeForge v1 — composition and one-page rendering

The final resume is **not** a direct rendering of whatever length the LLM happens to return. The structured generator first produces evidence-linked candidates; the composition stage selects additional entry-owned material; the fitter determines what can be presented legibly on one page.

## Data flow and responsibility

```text
generation/resume_generator.py
  TailoredResume with validated evidence-linked bullets
        |
generation/composition.py
  select career entries and top up entry-owned, validated bullet candidates
        |
rendering/latex_renderer.py
  produce ATS-oriented, one-column LaTeX (no LLM deciding page layout)
        |
rendering/page_fit.py
  compile trial PDF -> count pages -> adjust spacing / optional extras
        |
output/tailored_resume.{json,tex,pdf} + output/layout_report.json
```

## Content contract: `generation/composition.py`

Current settings: `EXPERIENCE_LIMIT=3`, `PROJECT_LIMIT=2`, `EXPERIENCE_TARGET=3` bullets per selected job and `PROJECT_TARGET=2` per selected project. The current implementation takes entries in canonical profile order; it is not a global JD-based entity-ranking policy.

For each entry, composition keeps accepted bullets only when every referenced evidence card has the entry's exact `entity_id` and `entity_type`. It then scores remaining eligible cards, generates concise candidate bullets and calls the claim validator. It limits extra generation attempts and applies a conservative text-similarity check. It can fall short of target; an entry with no validated achievement bullet is omitted with a warning. Existing bullets may still repeat the same accomplishment in different wording.

The skills list is selected from canonical profile skills, documented project technologies and evidence-card skill labels. The formatter sorts them into labeled rows; unmatched labels fall into `Other`. This is **not a technical-skill ontology**: manually remove vague actions, project features and incorrectly inferred skills before applying.

## Template: `rendering/latex_renderer.py`

The template is single-column and designed for text extraction, with a contact header, optional short summary, categorized technical skills, professional experience, projects, education and certifications. It escapes LaTeX-sensitive characters and removes *wrapping* Markdown emphasis from profile fields.

The renderer accepts `paper="letter"` or `paper="a4"` in its Python API; `run_pipeline.py` currently hard-codes `letter`. Four density presets define bounded font size, leading, margins and spacing: `expanded`, `comfortable`, `normal` and `compact`. Do not increase density by introducing unsupported resume text.

The template needs a working `pdflatex` and its LaTeX packages. `compile_pdf` raises with compiler output if compilation fails; it does not silently call a missing PDF a success.

## Fitting: `rendering/page_fit.py`

`render_one_page` deep-copies the structured resume and tries `normal` and `compact` PDF builds. For a one-page result with substantial measurable unused space, it also attempts `expanded` / `comfortable` spacing. If content overflows, it removes only optional extra bullets, optional summary or low-priority skills according to the fitter's rules and tries again.

It **never invents bullet text to occupy blank space** and will not delete an entry's sole validated bullet just to meet the page count. If no legal layout remains, it raises an exception. A completed run therefore has a checked **one-page PDF**, but not necessarily a page filled edge to edge.

`pypdf` checks the physical page count. The optional `fitz`/PyMuPDF library estimates the bottommost text position; when unavailable, `estimated_bottom_usage` is `null`. This number estimates vertical use, not visual quality or substantive completeness.

## Outputs and diagnostics

| Output | Meaning |
| --- | --- |
| `tailored_resume.json` | **Fitted** structured resume after any overflow trimming, with source evidence IDs on bullets. |
| `tailored_resume.tex` | LaTeX document that compiled to the final PDF; evidence IDs appear in comments. |
| `tailored_resume.pdf` | Final compiled one-page PDF. |
| `layout_report.json` | `page_count`, `density`, experience/project entry and bullet counts, optional `estimated_bottom_usage`, `warnings`. |

Examples of legitimate warnings include fewer verified bullets than the target, an entity omitted for lack of accepted evidence, optional content removed due to overflow, or insufficient verifiable content to fill the page. **Do not turn a warning into a fabricated achievement.**

## Limitations and next steps

A one-page check establishes only physical page count. It does not guarantee excellent wording, accurate skills, sufficient source evidence, employer eligibility, ATS compatibility in every system, or a submission-ready document. Review the PDF and evidence-linked JSON before use. Improvements to consider after v1 include true achievement-level deduplication, a curated technical-skills taxonomy and quantitative validation of years-of-experience requirements.

For setup see [SETUP.md](SETUP.md); for the CLI and input format see [RUN.md](RUN.md); for the full system see [ARCHITECTURE.md](ARCHITECTURE.md).
