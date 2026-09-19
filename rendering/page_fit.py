"""Compile, count PDF pages, apply bounded spacing/trimming, report page utilization.

No content is invented and no section is silently dropped to satisfy a page count.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from pypdf import PdfReader

from generation.resume_generator import TailoredResume
from rendering.latex_renderer import ResumeHeader, render_latex, compile_pdf


@dataclass
class PageReport:
    page_count: int
    density: str
    experience_entries: int
    project_entries: int
    experience_bullets: int
    project_bullets: int
    estimated_bottom_usage: float | None
    warnings: list[str]


def measure_pdf(path: Path) -> tuple[int, float | None]:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    usage = None
    # PyMuPDF is OPTIONAL; pypdf is already a ResumeForge dependency.
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            page = doc[0]
            rectangles = [block[:4] for block in page.get_text("blocks")
                          if str(block[4]).strip()]
            if rectangles:
                usage = round(max(rect[3] for rect in rectangles) / page.rect.height, 3)
    except (ImportError, AttributeError, RuntimeError, ValueError):
        pass
    return page_count, usage


def _remove_extra(resume: TailoredResume) -> str | None:
    # Don't erase a candidate's only validated bullet or silently drop entities.
    for entries in (resume.projects, resume.experience):
        for entry in reversed(entries):
            if len(entry.bullets) > 1:
                entry.bullets.pop()
                return entry.entry_id
    if resume.summary:
        resume.summary = []
        return "optional summary"
    if len(resume.skills) > 8:
        resume.skills = resume.skills[:-2]
        return "two low-priority skills"
    return None


def render_one_page(resume: TailoredResume, header: ResumeHeader,
                    output_dir: Path, *, paper: str = "letter",
                    initial_warnings: list[str] | None = None) -> tuple[TailoredResume, PageReport]:
    """Render a verified one-page PDF. Raise if content cannot fit legibly.

    Writes output/tailored_resume.tex, .pdf and output/layout_report.json.
    Input `resume` is deep-copied, because fitting may drop optional extras.
    """
    warnings = list(initial_warnings or [])
    fitted = resume.model_copy(deep=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = ("normal", "compact")
    with tempfile.TemporaryDirectory(prefix="resumeforge_fit_") as scratch:
        trial = Path(scratch) / "trial.tex"
        for attempt in range(32):
            for density in options:
                render_latex(fitted, header, trial, density=density, paper=paper)
                pdf = compile_pdf(trial)
                if pdf is None:
                    raise RuntimeError("pdflatex is required for verified one-page output")
                count, usage = measure_pdf(pdf)
                if count == 1:
                    # Expand type and spacing when there is substantial empty space,
                    # never with filler text or invented achievements.
                    if usage is not None and usage < 0.80 and density == "normal":
                        for candidate_density in ("expanded", "comfortable"):
                            render_latex(fitted, header, trial, density=candidate_density, paper=paper)
                            candidate_pdf = compile_pdf(trial)
                            candidate_count, candidate_usage = measure_pdf(candidate_pdf)
                            if candidate_count == 1:
                                density, usage = candidate_density, candidate_usage
                                break
                    destination_tex = output_dir / "tailored_resume.tex"
                    render_latex(fitted, header, destination_tex, density=density, paper=paper)
                    result_pdf = compile_pdf(destination_tex)
                    confirmed_count, confirmed_usage = measure_pdf(result_pdf)
                    if confirmed_count != 1:
                        raise RuntimeError("Final PDF changed page count during final compilation")
                    if confirmed_usage is not None and confirmed_usage < 0.72:
                        warnings.append("Page remains underfilled: add verified, nonredundant achievements to the corpus")
                    if not fitted.experience:
                        warnings.append("No experience entry contains a verified bullet")
                    if not fitted.projects:
                        warnings.append("No project entry contains a verified bullet")
                    report = PageReport(
                        page_count=1, density=density,
                        experience_entries=len(fitted.experience),
                        project_entries=len(fitted.projects),
                        experience_bullets=sum(len(e.bullets) for e in fitted.experience),
                        project_bullets=sum(len(e.bullets) for e in fitted.projects),
                        estimated_bottom_usage=confirmed_usage,
                        warnings=warnings,
                    )
                    (output_dir / "layout_report.json").write_text(
                        json.dumps(asdict(report), indent=2), encoding="utf-8")
                    return fitted, report
            changed = _remove_extra(fitted)
            if changed is None:
                raise RuntimeError(
                    "Verified one-page layout is not possible without deleting required content "
                    "or reducing font size below the configured minimum. "
                    "Use a two-page policy or reduce corpus selection explicitly."
                )
            warnings.append(f"One-page overflow: removed extra content from {changed}")
    raise RuntimeError("Page fitting exceeded attempt budget")
