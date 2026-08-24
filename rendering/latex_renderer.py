from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from generation.resume_generator import (
    TailoredResume,
)


class ResumeHeader(BaseModel):

    name: str

    email: str | None = None
    phone: str | None = None
    location: str | None = None

    linkedin: str | None = None
    github: str | None = None


LATEX_ESCAPE = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(
    text: str,
) -> str:

    result = []

    for char in text:

        result.append(
            LATEX_ESCAPE.get(
                char,
                char,
            )
        )

    return "".join(result)


def render_latex(
    resume: TailoredResume,
    header: ResumeHeader,
    output_file: Path,
) -> Path:

    contact_parts = [
        header.email,
        header.phone,
        header.location,
        header.linkedin,
        header.github,
    ]

    contact_parts = [
        latex_escape(part)
        for part in contact_parts
        if part
    ]

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{enumitem}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{titlesec}",
        r"\setlength{\parindent}{0pt}",
        r"\setlist[itemize]{leftmargin=*,nosep}",
        r"\titleformat{\section}"
        r"{\large\bfseries}{}{0em}{}"
        r"[\titlerule]",
        r"\begin{document}",
        r"\begin{center}",
        (
            r"{\LARGE\textbf{"
            + latex_escape(
                header.name
            )
            + r"}}"
        ),
    ]

    if contact_parts:

        lines.append(
            r"\\[3pt]"
            + " $|$ ".join(
                contact_parts
            )
        )

    lines.extend(
        [
            r"\end{center}",
            "",
        ]
    )

    if resume.summary:

        lines.append(
            r"\section*{Summary}"
        )

        summary_text = " ".join(
            item.text
            for item
            in resume.summary
        )

        lines.append(
            latex_escape(
                summary_text
            )
        )

        lines.append("")

    if resume.skills:

        lines.append(
            r"\section*{Technical Skills}"
        )

        lines.append(
            latex_escape(
                ", ".join(
                    resume.skills
                )
            )
        )

        lines.append("")

    section_titles = {
        "experience":
            "Experience",

        "projects":
            "Projects",

        "education":
            "Education",

        "certifications":
            "Certifications",
    }

    for section_name, bullets in (
        resume.sections.items()
    ):

        if not bullets:
            continue

        title = section_titles.get(
            section_name,
            section_name.title(),
        )

        lines.append(
            rf"\section*{{"
            f"{latex_escape(title)}"
            r"}"
        )

        grouped: dict[
            str,
            list,
        ] = {}

        for bullet in bullets:

            label = (
                bullet.source_labels[0]
                if bullet.source_labels
                else "Selected Experience"
            )

            grouped.setdefault(
                label,
                [],
            ).append(
                bullet
            )

        for label, entries in (
            grouped.items()
        ):

            lines.append(
                r"\textbf{"
                + latex_escape(label)
                + r"}"
            )

            lines.append(
                r"\begin{itemize}"
            )

            for entry in entries:

                evidence_comment = (
                    "% Evidence: "
                    + ", ".join(
                        entry.evidence_ids
                    )
                )

                lines.append(
                    evidence_comment
                )

                lines.append(
                    r"\item "
                    + latex_escape(
                        entry.text
                    )
                )

            lines.append(
                r"\end{itemize}"
            )

            lines.append("")

    lines.append(
        r"\end{document}"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_file


def compile_pdf(
    tex_file: Path,
) -> Path | None:

    executable = shutil.which(
        "pdflatex"
    )

    if executable is None:
        return None

    subprocess.run(
        [
            executable,
            "-interaction=nonstopmode",
            tex_file.name,
        ],
        cwd=tex_file.parent,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    return tex_file.with_suffix(
        ".pdf"
    )