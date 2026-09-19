"""ATS-oriented one-column ResumeForge LaTeX template (one-page fitter owns policy)."""
from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from generation.resume_generator import TailoredResume


class ResumeHeader(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None


LATEX_ESCAPE = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def clean_text(value: str | None) -> str:
    """Remove only *wrapping* Markdown emphasis from profile/template inputs."""
    value = str(value or "").strip()
    for wrapper in ("**", "__", "*"):
        if value.startswith(wrapper) and value.endswith(wrapper) and len(value) > 2 * len(wrapper):
            value = value[len(wrapper):-len(wrapper)].strip()
    return re.sub(r"\s+", " ", value)


def latex_escape(text: str | None) -> str:
    return "".join(LATEX_ESCAPE.get(char, char) for char in clean_text(text))


def date_label(text: str | None) -> str:
    value = clean_text(text)
    if not value:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}", value):
        try:
            return datetime.strptime(value, "%Y-%m").strftime("%b %Y")
        except ValueError:
            return value
    return value


def _section(lines: list[str], name: str):
    lines.append(r"\section*{" + latex_escape(name) + "}")


def _bullets(lines, bullets):
    if not bullets:
        return
    lines.append(r"\begin{itemize}")
    for bullet in bullets:
        if not bullet.validated:
            continue
        ids = ", ".join(bullet.evidence_ids)
        lines.append("% Evidence: " + ids.replace("\n", " "))
        lines.append(r"\item " + latex_escape(bullet.text))
    lines.append(r"\end{itemize}")


GROUPS = {
    "Languages": {"python", "sql", "c++", "c", "c#", "java", "javascript", "typescript", "bash", "powershell", "html", "css"},
    "Frameworks & Data": {"fastapi", "react", "node.js", "next.js", "pandas", "pydantic", "postgresql", "mysql", "sql server", "mongodb", "bm25", "rag", "sentence-transformers", "llama.cpp", "rest apis", "rest api"},
    "Cloud, DevOps & Testing": {"docker", "github actions", "git", "linux", "google cloud run", "gcp", "aws", "azure", "pytest", "selenium", "playwright", "junit", "cloudwatch", "splunk", "la tex", "latex"},
}


def group_skills(skills: list[str]) -> list[tuple[str, list[str]]]:
    result = {name: [] for name in GROUPS}
    other = []
    for item in skills:
        label = clean_text(item)
        if not label:
            continue
        key = label.casefold()
        found = False
        for category, terms in GROUPS.items():
            if key in terms:
                result[category].append(label)
                found = True
                break
        if not found:
            other.append(label)
    groups = [(label, members) for label, members in result.items() if members]
    if other:
        groups.append(("Other", other))
    return groups


def render_latex(resume: TailoredResume, header: ResumeHeader,
                 output_file: Path, *, density: str = "normal", paper: str = "letter") -> Path:
    if density not in {"expanded", "comfortable", "normal", "compact"}:
        raise ValueError(f"Unknown density: {density}")
    if paper not in {"letter", "a4"}:
        raise ValueError(f"Unknown paper size: {paper}")
    presets = {
        "expanded": (0.73, "10.5pt", "13.2pt", "10pt", "5pt", "4.5pt"),
        "comfortable": (0.65, "10.0pt", "11.6pt", "7pt", "4pt", "3.0pt"),
        "normal": (0.64, "9.6pt", "11.0pt", "5pt", "3pt", "2.2pt"),
        "compact": (0.59, "9.3pt", "10.5pt", "4pt", "2pt", "1.6pt"),
    }
    margin, font, leading, section_before, section_after, item_sep = presets[density]
    entry_gap = "4pt" if density == "expanded" else "1pt"
    lines = [
        (r"\documentclass[10pt," + ("letterpaper" if paper == "letter" else "a4paper") + r"]{article}"),
        rf"\usepackage[margin={margin}in]{{geometry}}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{enumitem}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{titlesec}",
        r"\usepackage{microtype}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0pt}",
        rf"\setlist[itemize]{{leftmargin=1.15em,topsep=1.1pt,partopsep=0pt,parsep=0pt,itemsep={item_sep}}}",
        r"\titleformat{\section}{\bfseries\normalsize}{}{0pt}{}[\vspace{-4pt}\titlerule]",
        rf"\titlespacing*{{\section}}{{0pt}}{{{section_before}}}{{{section_after}}}",
        r"\begin{document}",
        rf"\fontsize{{{font}}}{{{leading}}}\selectfont",
        r"\begin{center}",
        r"{\fontsize{15}{16}\selectfont\bfseries " + latex_escape(header.name) + r"}\par",
    ]
    contact_parts = [latex_escape(part) for part in (
        header.email, header.phone, header.location, header.linkedin, header.github
    ) if clean_text(part)]
    if contact_parts:
        lines.append(r"\vspace{2pt}{\small " + r" \textbar{} ".join(contact_parts) + r"}\par")
    lines.append(r"\end{center}")
    lines.append(r"\vspace{-9pt}")
    if resume.summary:
        _section(lines, "Summary")
        lines.append(latex_escape(" ".join(b.text for b in resume.summary if b.validated)))
    if resume.skills:
        _section(lines, "Technical Skills")
        # Three/four short labeled rows, rather than one long unstructured skill dump.
        for label, members in group_skills(resume.skills):
            lines.append(r"\textbf{" + latex_escape(label) + r":} " + latex_escape(", ".join(members)) + r"\par")
    if resume.experience:
        _section(lines, "Professional Experience")
        for entry in resume.experience:
            dates = " -- ".join(filter(None, [date_label(entry.start_date), date_label(entry.end_date)]))
            heading = r"\textbf{" + latex_escape(entry.title) + "}"
            if dates:
                heading += r"\hfill " + latex_escape(dates)
            lines.append(heading + r"\par")
            employer = clean_text(entry.company)
            if entry.location:
                employer += ", " + clean_text(entry.location)
            lines.append(r"\textit{" + latex_escape(employer) + r"}\par")
            _bullets(lines, entry.bullets)
            lines.append(r"\vspace{" + entry_gap + r"}")
    if resume.projects:
        _section(lines, "Projects")
        for entry in resume.projects:
            heading = r"\textbf{" + latex_escape(entry.name) + r"}"
            if entry.date:
                heading += r"\hfill " + latex_escape(date_label(entry.date))
            lines.append(heading + r"\par")
            if entry.technologies:
                lines.append(r"\textit{" + latex_escape(", ".join(entry.technologies[:7])) + r"}\par")
            _bullets(lines, entry.bullets)
            lines.append(r"\vspace{" + entry_gap + r"}")
    if resume.education:
        _section(lines, "Education")
        for entry in resume.education:
            degree = ", ".join(filter(None, [entry.degree, entry.field]))
            heading = r"\textbf{" + latex_escape(degree) + r"}"
            if entry.graduation_date:
                heading += r"\hfill " + latex_escape(date_label(entry.graduation_date))
            lines.append(heading + r"\par")
            school = clean_text(entry.institution)
            if entry.location:
                school += ", " + clean_text(entry.location)
            lines.append(latex_escape(school) + r"\par")
    if resume.certifications:
        _section(lines, "Certifications")
        for entry in resume.certifications:
            value = ", ".join(filter(None, [entry.name, entry.issuer, entry.date]))
            lines.append(latex_escape(value) + r"\par")
    lines.append(r"\end{document}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file


def compile_pdf(tex_file: Path) -> Path | None:
    compiler = shutil.which("pdflatex")
    if compiler is None:
        return None
    process = subprocess.run(
        [compiler, "-halt-on-error", "-file-line-error", "-interaction=nonstopmode", tex_file.name],
        cwd=tex_file.parent, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if process.returncode != 0:
        raise RuntimeError("pdflatex failed:\n" + process.stdout[-5000:])
    pdf = tex_file.with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError("pdflatex succeeded but the output PDF is missing")
    return pdf
