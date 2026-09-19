# ResumeForge v1 architecture

ResumeForge separates **source facts**, **candidate structure**, **job relevance**, **generated language**, and **page layout**. The LLM writes candidate claims; Python owns entity identities, resume structure and the physical one-page policy.

## Pipeline

```text
corpus/raw/ (canonical Markdown plus optional unstructured PDF/MD/TXT)
  -> ingestion/parser.py + ingestion/canonical_parser.py
       ParsedDocument(entity_id, entity_type, metadata, source_file, text)
  -> generation/candidate_profile.py (contact, skills, experience, projects,
       education, certifications)
  -> ingestion/chunker.py (entity-bounded chunks)
  -> ingestion/evidence_extractor.py (cached, sourced evidence cards)
  -> retrieval/{bm25,semantic,hybrid,reranker}.py
       + jd/requirement_extractor.py + matching/{matcher,evidence_map}.py
  -> generation/{planner,bullet_generator,resume_generator}.py
       + validation/claim_validator.py
  -> generation/composition.py
  -> rendering/latex_renderer.py + rendering/page_fit.py
  -> output/{tailored_resume.json,tailored_resume.tex,
             tailored_resume.pdf,layout_report.json}
```

The terminal reports ten stages: load model; parse corpus; reconstruct profile; chunk; extract evidence; build retrieval; analyze JD; match evidence; generate structured resume; compose and render one-page output.

## 1. Canonical corpus and entity ownership

The preferred source contract is `corpus/raw/profile.md`, `experience.md`, and `projects.md`. The canonical parser splits one physical file into separate semantic records: each job, project, education entry and the general profile receive a stable `entity_id` and `entity_type`. An experience heading may be `## exp_example_company_2024`; a project heading may be `## project_example`.

`ingestion/parser.py` also accepts arbitrary PDF, Markdown and TXT as unstructured fallback sources, but those records do **not** necessarily have known employer/project ownership. Prefer canonical Markdown for reliable attribution; the LLM fallback is not a replacement for explicit source structure.

Chunk and evidence records inherit source metadata. Resume-entry-specific bullet planning and composition reject evidence that belongs to another entity; general profile evidence must not be presented as an accomplishment at a named employer. Entity provenance prevents cross-entry attribution, but it does not guarantee that an LLM-extracted fact was faithfully understood within its own chunk.

## 2. Retrieval, generation and validation

`ingestion/evidence_extractor.py` converts text chunks into evidence cards. Its model-aware cache skips unchanged extractions; switching models can require re-extracting evidence, so evidence-card counts can differ between runs.

`jd/requirement_extractor.py` extracts a normalized requirement list. The retrieval/matching modules rank and map relevant candidate evidence. `generation/planner.py` works with a bounded evidence context and proposes bullets tied to entry IDs and evidence IDs. `generation/bullet_generator.py` writes individual claims; `validation/claim_validator.py` accepts/rejects them using the referenced evidence. The structured `TailoredResume` contains entry metadata and evidence-linked bullet records.

**Important:** a model's STRONG/PARTIAL/NONE judgment is an evidence-matching output, **not an eligibility or hiring-probability assessment**. v1 does not reliably validate experience-duration thresholds or every “hands-on” requirement against a complete timeline. A supported bullet may still be awkward, repetitive, or originate from incorrectly extracted source evidence.

## 3. Deterministic composition

`generation/composition.py` runs **after** structured generation. It retains entry metadata from the canonical candidate profile, reuses valid entry-owned bullets, and attempts additional bullets from evidence cards belonging to the same entry.

Current initial budgets:

| Selection | Policy |
| --- | --- |
| Experience entries | Up to 3, in canonical profile order |
| Bullets per experience | Target 3, subject to validated evidence |
| Project entries | Up to 2, in canonical profile order |
| Bullets per project | Target 2, subject to validated evidence |
| Summary | Optional; at most 1 paragraph/bullet |
| Skills | Ranked from profile, project technologies and extracted skill labels; up to 16 |

The composition stage does not invent achievements to meet a target. An entry with no accepted bullet is omitted and a warning is produced. Duplicate filtering is conservative and cannot guarantee that paraphrases of the same achievement are removed. Skill labels from LLM-extracted evidence must be reviewed manually.

## 4. Rendering and page fitting

`rendering/latex_renderer.py` writes the one-column LaTeX template, cleans wrapping Markdown emphasis, escapes LaTeX-sensitive characters, and renders the contact header, optional summary, categorized skills, experience, projects, education and certifications. The renderer's paper options are `letter` and `a4`; the current CLI pipeline passes `paper="letter"`.

`rendering/page_fit.py` compiles trial PDFs with `pdflatex`, counts pages using `pypdf`, tries spacing presets (`normal`, `compact`, and expansion when measurable), and removes optional excess content if necessary. A one-page PDF is required for success. The fitter will not remove an entry's last validated bullet merely to meet the page count; if it cannot fit, it raises an error. It returns the **fitted** resume, which is saved as JSON alongside the final TeX/PDF.

`layout_report.json` records the chosen density, page count, number of entries/bullets, optional estimated bottom-page utilization, and warnings. Vertical utilization requires the optional `fitz`/PyMuPDF dependency; without it, page count is still checked but bottom-page utilization can be `null`. A full page is a layout goal, not permission to invent experience.

See [RENDERING.md](RENDERING.md) for details.

## Trust boundaries and limitations

- Candidate source documents are the source of truth; LLM output is not.
- Canonical ownership, evidence-ID linkage, LLM claim validation and PDF fitting address **different** failure modes.
- The final one-page PDF is not automatically “submission-ready”: verify facts and source attribution, skills, dates, relevance, readable rendering, and absence of repeated accomplishments.
- ResumeForge v1 generates a resume only; job scraping and application submission belong to the larger Autonomous Job Applier project.
