"""Deterministic ResumeForge content policy, run after the existing resume generator.

The LLM writes individual grounded bullets; this module decides *how many* and
*where*. It never creates achievements, employers, dates, or metrics itself.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from generation.bullet_generator import generate_bullet
from generation.planner import PlannedBullet
from generation.resume_generator import (
    ResumeBulletRecord, ResumeExperienceEntry, ResumeProjectEntry, TailoredResume,
    source_label,
)
from validation.claim_validator import validate_bullet

# Initial one-page content contract. Page fitter may remove extras if necessary.
EXPERIENCE_TARGET = 3
PROJECT_TARGET = 2
EXPERIENCE_LIMIT = 3
PROJECT_LIMIT = 2
MAX_GENERATION_ATTEMPTS_PER_ENTITY = 5

# Evidence is assigned to an entity by the canonical parser, never by the LLM.
ACTION = re.compile(
    r"\b(built|developed|designed|implemented|automated|tested|deployed|"
    r"integrated|optimized|created|maintained|investigated|configured|"
    r"improved|led|supported|documented|analyzed|engineered|reduced|"
    r"authored|wrote|delivered)\b", re.IGNORECASE,
)
FILLER = re.compile(r"\b(detail.oriented|passionate|demonstrat(?:ed|ing)\s+"
                    r"(?:technical\s+)?proficiency|familiar with|highly skilled)\b", re.I)


def _similar(a: str, b: str) -> bool:
    """Conservative semantic-overlap filter: also blocks verbatim repetitions."""
    ta = set(re.findall(r"[a-z0-9]+", a.casefold()))
    tb = set(re.findall(r"[a-z0-9]+", b.casefold()))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.72


def _match_priority(evidence_map: Any) -> tuple[dict[str, int], dict[str, list[str]]]:
    points: dict[str, int] = defaultdict(int)
    requirements: dict[str, list[str]] = defaultdict(list)
    weights = {"strong": 8, "partial": 4, "none": 0}
    for result in evidence_map.requirements:
        weight = weights.get(str(result.match_strength).casefold(), 0)
        for ev_id in result.evidence_ids:
            points[ev_id] = max(points[ev_id], weight)
            if weight and result.requirement_id not in requirements[ev_id]:
                requirements[ev_id].append(result.requirement_id)
    return points, requirements


def _card_score(evidence_id: str, card: dict, priority: dict[str, int]) -> tuple:
    claim = str(card.get("claim") or "")
    category = str(card.get("category") or "").casefold()
    metrics = card.get("metrics") or []
    category_weight = {"achievement": 4, "experience": 3, "project": 3,
                       "responsibility": 2, "skill": -2, "other": -3}.get(category, 0)
    return (priority.get(evidence_id, 0) + (4 if metrics else 0)
            + category_weight + (2 if ACTION.search(claim) else 0)
            - (5 if FILLER.search(claim) else 0),
            min(len(claim), 180), evidence_id)


def _eligible(evidence_id: str, card: dict, entry_id: str, entity_type: str) -> bool:
    claim = str(card.get("claim") or "").strip()
    return (
        card.get("entity_id") == entry_id
        and card.get("entity_type") == entity_type
        and len(claim) >= 26
        and not FILLER.search(claim)
        and str(card.get("category") or "") not in {"education", "certification", "skill"}
    )


def _keep_existing(bullets: list[ResumeBulletRecord], entry_id: str,
                   entity_type: str, evidence_lookup: dict) -> list[ResumeBulletRecord]:
    accepted = []
    for bullet in bullets:
        if len(bullet.text.split()) > 43 or FILLER.search(bullet.text):
            continue
        if not bullet.evidence_ids or any(
            (card := evidence_lookup.get(evidence_id)) is None
            or card.get("entity_id") != entry_id
            or card.get("entity_type") != entity_type
            for evidence_id in bullet.evidence_ids
        ):
            continue
        if any(_similar(bullet.text, previous.text) for previous in accepted):
            continue
        accepted.append(bullet)
    return accepted


def _generate_for_entry(llm, validator_llm, section: str, entry_id: str,
                        target: int, initial: list[ResumeBulletRecord],
                        evidence_lookup: dict, evidence_map, priority: dict[str, int],
                        requirements: dict[str, list[str]], warnings: list[str]):
    kind = "experience" if section == "experience" else "project"
    accepted = _keep_existing(initial, entry_id, kind, evidence_lookup)
    if len(accepted) >= target:
        return accepted[:target]

    candidates = sorted(
        [(ev_id, card) for ev_id, card in evidence_lookup.items()
         if _eligible(ev_id, card, entry_id, kind)],
        key=lambda pair: _card_score(pair[0], pair[1], priority), reverse=True,
    )
    used_ids = {ev_id for bullet in accepted for ev_id in bullet.evidence_ids}
    calls = 0
    for ev_id, card in candidates:
        if len(accepted) >= target or calls >= MAX_GENERATION_ATTEMPTS_PER_ENTITY:
            break
        if ev_id in used_ids or any(_similar(card["claim"], b.text) for b in accepted):
            continue
        calls += 1
        plan = PlannedBullet(
            section=section, entry_id=entry_id,
            objective=("Write ONE specific action and result from this single evidence claim, "
                       "using at most 28 words. No generic skill list. "
                       "Do not introduce new facts."),
            requirement_ids=requirements.get(ev_id, [])[:2],
            evidence_ids=[ev_id],
            plan_id="compose_" + hashlib.sha256(f"{entry_id}/{ev_id}".encode()).hexdigest()[:12],
        )
        feedback = None
        for attempt in range(2):
            try:
                generated = generate_bullet(llm, plan, evidence_lookup, feedback=feedback)
                if len(generated.text.split()) > 38 or FILLER.search(generated.text):
                    feedback = "Use no more than 28 words, one specific action, and no generic skill list."
                    continue
                verdict = validate_bullet(validator_llm, generated, evidence_lookup)
                if verdict.supported and not any(
                    _similar(generated.text, prev.text) for prev in accepted
                ):
                    accepted.append(ResumeBulletRecord(
                        section=section, entry_id=entry_id, text=generated.text,
                        evidence_ids=[ev_id], requirement_ids=plan.requirement_ids,
                        source_labels=[source_label(card)], validated=True,
                    ))
                    used_ids.add(ev_id)
                    break
                feedback = verdict.reason + "; " + "; ".join(verdict.unsupported_claims)
            except (RuntimeError, ValueError, KeyError) as error:
                warnings.append(f"{entry_id}: bullet candidate failed: {type(error).__name__}")
                break
    if len(accepted) < target:
        warnings.append(f"{entry_id}: {len(accepted)}/{target} verified bullets available")
    return accepted[:target]


def _pick_skills(profile, resume, analysis, evidence_lookup: dict) -> list[str]:
    """Skills must come from the canonical profile or entity-linked evidence."""
    inventory = []
    for value in profile.skills:
        if isinstance(value, str) and 1 <= len(value.strip()) <= 40:
            inventory.append(value.strip())
    for project in profile.projects:
        inventory.extend(str(s).strip() for s in project.technologies if s)
    for card in evidence_lookup.values():
        inventory.extend(str(s).strip() for s in card.get("skills", []) if s)
    jd_terms = " ".join([r.text + " " + " ".join(r.skills) for r in analysis.requirements]).casefold()
    # Don't promote a JD term into a candidate skill unless it appears in inventory.
    unique: dict[str, str] = {}
    for value in inventory:
        if not value or len(value) > 42 or ":" in value or "\n" in value:
            continue
        unique.setdefault(value.casefold(), value)
    existing = {str(x).casefold() for x in resume.skills}
    def score(value):
        mentioned = re.search(r"(?<![\w])" + re.escape(value.casefold()) + r"(?![\w])", jd_terms)
        return (6 if mentioned else 0) + (2 if value.casefold() in existing else 0)
    return sorted(unique.values(), key=lambda value: (-score(value), value.casefold()))[:16]


def compose_resume(resume: TailoredResume, *, profile, analysis, evidence_map,
                   evidence_lookup: dict, generator_llm, validator_llm=None):
    """Top-up grounded bullets per entity; keep the core entry selection deterministic.

    Returns (new_resume, warnings). It does not fit the PDF; page_fit.py handles
    physical measurement and overflow without fabricating new text.
    """
    validator_llm = validator_llm or generator_llm
    warnings: list[str] = []
    priority, requirements = _match_priority(evidence_map)
    baseline = resume.model_copy(deep=True)
    current_experience = {entry.entry_id: entry for entry in baseline.experience}
    current_projects = {entry.entry_id: entry for entry in baseline.projects}
    baseline.experience = []
    baseline.projects = []
    # Include all experience when there are <=3; do not insert unsupported entries.
    for item in profile.experience[:EXPERIENCE_LIMIT]:
        existing = current_experience.get(item.entry_id)
        bullets = _generate_for_entry(
            generator_llm, validator_llm, "experience", item.entry_id,
            EXPERIENCE_TARGET, existing.bullets if existing else [],
            evidence_lookup, evidence_map, priority, requirements, warnings,
        )
        if bullets:
            baseline.experience.append(ResumeExperienceEntry(
                entry_id=item.entry_id, company=item.company, title=item.title,
                location=item.location, start_date=item.start_date,
                end_date=item.end_date, bullets=bullets,
            ))
        else:
            warnings.append(f"{item.entry_id}: omitted (no verified achievement evidence)")
    # Keep at most two project entries, in canonical order for this first version.
    for item in profile.projects[:PROJECT_LIMIT]:
        existing = current_projects.get(item.entry_id)
        bullets = _generate_for_entry(
            generator_llm, validator_llm, "projects", item.entry_id,
            PROJECT_TARGET, existing.bullets if existing else [],
            evidence_lookup, evidence_map, priority, requirements, warnings,
        )
        if bullets:
            baseline.projects.append(ResumeProjectEntry(
                entry_id=item.entry_id, name=item.name,
                technologies=item.technologies[:7], date=item.date, bullets=bullets,
            ))
        else:
            warnings.append(f"{item.entry_id}: omitted (no verified achievement evidence)")
    baseline.skills = _pick_skills(profile, baseline, analysis, evidence_lookup)
    # A summary that repeats experience consumes space without new information.
    if baseline.summary and any(
        _similar(baseline.summary[0].text, b.text)
        for entry in baseline.experience + baseline.projects for b in entry.bullets
    ):
        baseline.summary = []
    baseline.summary = baseline.summary[:1]
    return baseline, warnings
