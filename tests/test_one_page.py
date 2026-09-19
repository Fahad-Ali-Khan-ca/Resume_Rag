"""Offline integration smoke test. Run: python tests/test_one_page.py"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The smoke test replaces external model and GitHub project modules with stubs.
# Real project files are supplied by the user's working checkout.

def module(name):
    result = types.ModuleType(name)
    sys.modules[name] = result
    return result

local = module("local_llm")
local.LLM = type("LLM", (), {})
local.GenConfig = type("GenConfig", (), {"__init__": lambda self, **kwargs: None})

jd_pkg = module("jd")
jd_pkg.__path__ = []
jd = module("jd.schemas")
jd.JobDescriptionAnalysis = type("JobDescriptionAnalysis", (), {})

match_pkg = module("matching")
match_pkg.__path__ = []
matches = module("matching.evidence_map")
matches.EvidenceMap = type("EvidenceMap", (), {})

class EducationEntry(BaseModel):
    entry_id: str = "edu_1"
    degree: str = "BEng"
    field: str = "Software Engineering"
    institution: str = "Sample University"
    location: str | None = "Sample City"
    graduation_date: str | None = "2025"

class CertificationEntry(BaseModel):
    entry_id: str
    name: str
    issuer: str | None = None
    date: str | None = None

candidate = module("generation.candidate_profile")
candidate.CandidateProfile = type("CandidateProfile", (), {})
candidate.EducationEntry = EducationEntry
candidate.CertificationEntry = CertificationEntry

class PlannedBullet(BaseModel):
    section: str
    entry_id: str | None = None
    objective: str
    requirement_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    plan_id: str

planner = module("generation.planner")
planner.PlannedBullet = PlannedBullet
planner.ResumeSection = str
planner.plan_resume = lambda *args, **kwargs: None

bullet = module("generation.bullet_generator")
bullet.generate_bullet = lambda llm, plan, evidence_lookup, feedback=None: types.SimpleNamespace(
    section=plan.section, entry_id=plan.entry_id,
    text=evidence_lookup[plan.evidence_ids[0]]["claim"],
    evidence_ids=plan.evidence_ids, requirement_ids=plan.requirement_ids,
)
valid_pkg = module("validation")
valid_pkg.__path__ = []
validator = module("validation.claim_validator")
validator.validate_bullet = lambda llm, generated, evidence: types.SimpleNamespace(
    supported=generated.text == evidence[generated.evidence_ids[0]]["claim"],
    reason="", unsupported_claims=[],
)

# Pull in the exact structured TailoredResume schema shipped earlier.
import importlib.util
source = ROOT / "generation" / "resume_generator.py"
if not source.exists():
    raise RuntimeError('Run this test from a ResumeForge checkout with generation/resume_generator.py')
spec = importlib.util.spec_from_file_location("generation.resume_generator", source)
resume_mod = importlib.util.module_from_spec(spec)
sys.modules["generation.resume_generator"] = resume_mod
spec.loader.exec_module(resume_mod)

from generation.composition import compose_resume
from rendering.latex_renderer import ResumeHeader, render_latex
from rendering.page_fit import render_one_page, measure_pdf


def run():
    experience = [
        types.SimpleNamespace(entry_id="exp_a", company="Example A", title="Software Engineer Intern",
                              location="Example City", start_date="2024-05", end_date="2024-08"),
        types.SimpleNamespace(entry_id="exp_b", company="Example B", title="Software Developer Intern",
                              location="Example City", start_date="2025-01", end_date="2025-04"),
    ]
    projects = [
        types.SimpleNamespace(entry_id="project_rag", name="Document Search Project", date=None,
                              technologies=["Python", "FastAPI", "RAG"]),
        types.SimpleNamespace(entry_id="project_cloud", name="Cloud Application", date=None,
                              technologies=["Docker", "React", "Google Cloud Run"]),
    ]
    profile = types.SimpleNamespace(experience=experience, projects=projects,
                                    skills=["Python", "SQL", "Git", "Docker"],
                                    education=[EducationEntry()], certifications=[])
    claims = {
        "exp_a": [
            "Built Python backend services with documented REST endpoints for internal data processing.",
            "Developed automated test coverage with pytest for critical backend components.",
            "Investigated production errors using service logs and documented the root causes.",
        ],
        "exp_b": [
            "Built full-stack product features using React, TypeScript, and Node.js services.",
            "Automated a Docker deployment workflow with GitHub Actions and automated test gates.",
            "Documented API requirements and coordinated implementation with the product team.",
        ],
        "project_rag": [
            "Implemented a Python ingestion pipeline for PDF and Markdown source documents.",
            "Built a hybrid retrieval layer combining lexical ranking with semantic similarity.",
            "Added evidence-linked claim validation before generating final resume documents.",
        ],
        "project_cloud": [
            "Built a containerized Python backend for a cloud-hosted web application.",
            "Integrated a React frontend with backend API endpoints for application workflows.",
        ],
    }
    evidence = {}
    i = 0
    for entity, texts in claims.items():
        for text in texts:
            i += 1
            evidence[f"ev_{i}"] = {"evidence_id": f"ev_{i}", "entity_id": entity,
                                    "entity_type": "experience" if entity.startswith("exp") else "project",
                                    "source_file": "experience.md" if entity.startswith("exp") else "projects.md",
                                    "category": "achievement", "claim": text,
                                    "metrics": [], "skills": ["Python"]}
    evidence_map = types.SimpleNamespace(requirements=[types.SimpleNamespace(
        evidence_ids=list(evidence)[:3], match_strength="strong", requirement_id="req_1")])
    jd = types.SimpleNamespace(requirements=[types.SimpleNamespace(text="Python development and API design", skills=["Python"])])
    initial = resume_mod.TailoredResume(
        target_role="Software Engineer", skills=["Python"], education=profile.education,
        experience=[resume_mod.ResumeExperienceEntry(
            entry_id="exp_a", company="Example A", title="Software Engineer Intern",
            location="Example City", start_date="2024-05", end_date="2024-08",
            bullets=[resume_mod.ResumeBulletRecord(section="experience", entry_id="exp_a",
               text=claims["exp_a"][0], evidence_ids=["ev_1"], requirement_ids=["req_1"],
               source_labels=["Example A"], validated=True)])],
    )
    # Deliberate cross-entity injection must be removed from the final output.
    initial.projects.append(resume_mod.ResumeProjectEntry(
        entry_id="project_rag", name="Document Search Project", technologies=["Python"],
        bullets=[resume_mod.ResumeBulletRecord(section="projects", entry_id="project_rag",
           text=claims["exp_b"][1], evidence_ids=["ev_5"], requirement_ids=[],
           source_labels=["Example B"], validated=True)],
    ))
    result, warnings = compose_resume(initial, profile=profile, analysis=jd,
                                      evidence_map=evidence_map, evidence_lookup=evidence,
                                      generator_llm=object(), validator_llm=object())
    assert len(result.experience) == 2, "missing experience"
    assert len(result.projects) == 2, "missing project"
    assert sum(len(x.bullets) for x in result.experience) == 6
    assert sum(len(x.bullets) for x in result.projects) == 4
    assert all(evidence[ev]["entity_id"] == entry.entry_id
               for entry in result.experience + result.projects
               for b in entry.bullets for ev in b.evidence_ids), "cross-entity attribution"
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder)
        header = ResumeHeader(name="**Sample Candidate**", email="sample@example.test",
                              location="Example City", linkedin="linkedin.com/in/example")
        final, report = render_one_page(result, header, target, initial_warnings=warnings)
        assert report.page_count == 1
        assert (target / "tailored_resume.tex").exists()
        assert (target / "tailored_resume.pdf").exists()
        tex = (target / "tailored_resume.tex").read_text()
        assert "**Sample Candidate**" not in tex
        assert r"\textbf{Sample Candidate}" not in tex # Header uses bfseries, no Markdown
        assert r"\section*{Professional Experience}" in tex
        assert "Example A" in tex and "Example B" in tex
        assert "Document Search Project" in tex and "Cloud Application" in tex
        import shutil
        shutil.copy2(target / "tailored_resume.pdf", ROOT / "template_smoke_test.pdf")
        shutil.copy2(target / "tailored_resume.tex", ROOT / "template_smoke_test.tex")
        print(f"PASS: one page; experience={report.experience_bullets}, project={report.project_bullets}; density={report.density}; utilization={report.estimated_bottom_usage}")
        print("PASS: wrong-entity evidence dropped; verified entry coverage and header escaping")

if __name__ == "__main__":
    run()
