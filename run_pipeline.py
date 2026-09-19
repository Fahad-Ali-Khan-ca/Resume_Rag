from __future__ import annotations

import argparse
from pathlib import Path

from local_llm import load

from ingestion.parser import (
    parse_corpus,
    save_documents,
)

from ingestion.chunker import (
    load_documents,
    create_chunks,
    save_chunks,
)

from ingestion.evidence_extractor import (
    extract_corpus,
)

from retrieval.common import (
    load_evidence,
)

from retrieval.bm25 import (
    BM25Retriever,
)

from retrieval.semantic import (
    SemanticRetriever,
)

from retrieval.hybrid import (
    HybridRetriever,
)

from retrieval.reranker import (
    EvidenceReranker,
)

from jd.requirement_extractor import (
    extract_requirements,
)

from matching.matcher import (
    EvidenceMatcher,
)

from matching.evidence_map import (
    build_evidence_map,
)

from generation.candidate_profile import (
    extract_candidate_profile,
    save_candidate_profile,
)

from generation.resume_generator import (
    generate_resume,
    save_resume,
)
from generation.composition import compose_resume

from rendering.latex_renderer import (
    ResumeHeader,
    render_latex,
    compile_pdf,
)
from rendering.page_fit import render_one_page


OUTPUT_DIR = Path("output")


def run(
    model_name: str,
    jd_file: Path,
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    linkedin: str | None = None,
    github: str | None = None,
) -> None:

    # ---------------------------------------------------------
    # 1. Load LLM
    # ---------------------------------------------------------

    print("\n[1/10] Loading model")

    llm = load(model_name)

    print(f"Model: {llm.name}")

    # ---------------------------------------------------------
    # 2. Parse raw corpus
    # ---------------------------------------------------------

    print("\n[2/10] Parsing corpus")

    documents = parse_corpus()

    save_documents(documents)

    print(
        f"Parsed {len(documents)} document records"
    )

    # ---------------------------------------------------------
    # 3. Reconstruct candidate profile
    # ---------------------------------------------------------

    print(
        "\n[3/10] Reconstructing candidate profile"
    )

    profile = extract_candidate_profile(
        llm=llm,
        documents=documents,
    )

    save_candidate_profile(
        profile
    )

    print(
        "Profile: "
        f"{len(profile.experience)} experience, "
        f"{len(profile.projects)} projects, "
        f"{len(profile.education)} education, "
        f"{len(profile.certifications)} certifications"
    )

    # ---------------------------------------------------------
    # 4. Chunk documents
    # ---------------------------------------------------------

    print(
        "\n[4/10] Chunking documents"
    )

    loaded_documents = load_documents()

    chunks = create_chunks(
        loaded_documents
    )

    save_chunks(chunks)

    print(
        f"Created {len(chunks)} chunks"
    )

    # ---------------------------------------------------------
    # 5. Extract evidence
    # ---------------------------------------------------------

    print(
        "\n[5/10] Extracting evidence"
    )

    extract_corpus(
        llm=llm
    )

    # ---------------------------------------------------------
    # 6. Build retrieval system
    # ---------------------------------------------------------

    print(
        "\n[6/10] Building retrieval system"
    )

    evidence_cards = load_evidence()

    evidence_lookup = {
        card["evidence_id"]: card
        for card in evidence_cards
    }

    bm25 = BM25Retriever(
        evidence_cards
    )

    semantic = SemanticRetriever(
        evidence_cards
    )

    hybrid = HybridRetriever(
        bm25=bm25,
        semantic=semantic,
    )

    reranker = EvidenceReranker()

    matcher = EvidenceMatcher(
        retriever=hybrid,
        reranker=reranker,
    )

    print(
        f"Indexed {len(evidence_cards)} evidence cards"
    )

    # ---------------------------------------------------------
    # 7. Parse job description
    # ---------------------------------------------------------

    print(
        "\n[7/10] Analyzing job description"
    )

    job_description = (
        jd_file.read_text(
            encoding="utf-8"
        )
    )

    analysis = extract_requirements(
        llm=llm,
        job_description=job_description,
    )

    print(
        f"Role: {analysis.role_title}"
    )

    print(
        f"Requirements: "
        f"{len(analysis.requirements)}"
    )

    # ---------------------------------------------------------
    # 8. Match evidence
    # ---------------------------------------------------------

    print(
        "\n[8/10] Matching evidence"
    )

    candidate_sets = (
        matcher.match_all(
            analysis
        )
    )

    evidence_map = (
        build_evidence_map(
            llm=llm,
            analysis=analysis,
            candidate_sets=candidate_sets,
        )
    )

    for item in evidence_map.requirements:
        print(
            f"{item.match_strength.upper():7} "
            f"{item.requirement_text}"
        )

    # ---------------------------------------------------------
    # 9. Generate structured resume
    # ---------------------------------------------------------

    print(
        "\n[9/10] Generating structured resume"
    )

    resume = generate_resume(
        generator_llm=llm,
        validator_llm=llm,
        analysis=analysis,
        evidence_map=evidence_map,
        evidence_lookup=evidence_lookup,
        profile=profile,
    )

    # Compose enough verified material for the template before fitting the page.
    resume, composition_warnings = compose_resume(
        resume,
        profile=profile,
        analysis=analysis,
        evidence_map=evidence_map,
        evidence_lookup=evidence_lookup,
        generator_llm=llm,
        validator_llm=llm,
    )

    # ---------------------------------------------------------
    # 10. Render verified one-page LaTeX / PDF
    # ---------------------------------------------------------
    print("\n[10/10] Composing and rendering one-page resume")
    header = ResumeHeader(
        name=(name or profile.name or "Candidate"),
        email=(email or profile.email),
        phone=(phone or profile.phone),
        location=(location or profile.location),
        linkedin=(linkedin or profile.linkedin),
        github=(github or profile.github),
    )
    fitted_resume, layout_report = render_one_page(
        resume, header, OUTPUT_DIR, paper="letter",
        initial_warnings=composition_warnings,
    )
    save_resume(fitted_resume, OUTPUT_DIR / "tailored_resume.json")
    print(f"JSON: {OUTPUT_DIR / 'tailored_resume.json'}")
    print(f"LaTeX: {OUTPUT_DIR / 'tailored_resume.tex'}")
    print(f"PDF: {OUTPUT_DIR / 'tailored_resume.pdf'}")
    print(f"Layout: {OUTPUT_DIR / 'layout_report.json'}")
    print(f"Pages: {layout_report.page_count}; "
          f"Experience bullets: {layout_report.experience_bullets}; "
          f"Project bullets: {layout_report.project_bullets}")
    for warning in layout_report.warnings:
        print(f"  Layout warning: {warning}")

    print(
        "\nResumeForge complete."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an evidence-grounded "
            "resume for a job description."
        )
    )

    parser.add_argument(
        "--model",
        default="gemma-2b",
    )

    parser.add_argument(
        "--jd",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--name",
    )

    parser.add_argument(
        "--email",
    )

    parser.add_argument(
        "--phone",
    )

    parser.add_argument(
        "--location",
    )

    parser.add_argument(
        "--linkedin",
    )

    parser.add_argument(
        "--github",
    )

    args = parser.parse_args()

    run(
        model_name=args.model,
        jd_file=args.jd,
        name=args.name,
        email=args.email,
        phone=args.phone,
        location=args.location,
        linkedin=args.linkedin,
        github=args.github,
    )


if __name__ == "__main__":
    main()
