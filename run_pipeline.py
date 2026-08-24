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

from generation.resume_generator import (
    generate_resume,
    save_resume,
)

from rendering.latex_renderer import (
    ResumeHeader,
    render_latex,
    compile_pdf,
)


OUTPUT_DIR = Path("output")


def run(
    model_name: str,
    jd_file: Path,
    name: str,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    linkedin: str | None = None,
    github: str | None = None,
) -> None:

    # ---------------------------------------------------------
    # 1. Load LLM
    # ---------------------------------------------------------

    print("\n[1/9] Loading model")

    llm = load(model_name)

    print(f"Model: {llm.name}")


    # ---------------------------------------------------------
    # 2. Parse raw corpus
    # ---------------------------------------------------------

    print("\n[2/9] Parsing corpus")

    documents = parse_corpus()

    save_documents(documents)

    print(
        f"Parsed {len(documents)} document records"
    )


    # ---------------------------------------------------------
    # 3. Chunk documents
    # ---------------------------------------------------------

    print("\n[3/9] Chunking documents")

    loaded_documents = load_documents()

    chunks = create_chunks(
        loaded_documents
    )

    save_chunks(chunks)

    print(
        f"Created {len(chunks)} chunks"
    )


    # ---------------------------------------------------------
    # 4. Extract evidence
    # ---------------------------------------------------------

    print("\n[4/9] Extracting evidence")

    extract_corpus(
        llm=llm
    )


    # ---------------------------------------------------------
    # 5. Build retrieval system
    # ---------------------------------------------------------

    print("\n[5/9] Building retrieval system")

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
    # 6. Parse job description
    # ---------------------------------------------------------

    print("\n[6/9] Analyzing job description")

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
    # 7. Match evidence
    # ---------------------------------------------------------

    print("\n[7/9] Matching evidence")

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
    # 8. Generate resume
    # ---------------------------------------------------------

    print("\n[8/9] Generating resume")

    resume = generate_resume(
        generator_llm=llm,
        validator_llm=llm,
        analysis=analysis,
        evidence_map=evidence_map,
        evidence_lookup=evidence_lookup,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_output = (
        OUTPUT_DIR
        / "tailored_resume.json"
    )

    save_resume(
        resume,
        json_output,
    )

    print(
        f"JSON: {json_output}"
    )


    # ---------------------------------------------------------
    # 9. Render LaTeX / PDF
    # ---------------------------------------------------------

    print("\n[9/9] Rendering resume")

    header = ResumeHeader(
        name=name,
        email=email,
        phone=phone,
        location=location,
        linkedin=linkedin,
        github=github,
    )

    tex_file = render_latex(
        resume=resume,
        header=header,
        output_file=(
            OUTPUT_DIR
            / "tailored_resume.tex"
        ),
    )

    print(
        f"LaTeX: {tex_file}"
    )

    pdf_file = compile_pdf(
        tex_file
    )

    if pdf_file:
        print(
            f"PDF: {pdf_file}"
        )
    else:
        print(
            "PDF not compiled because "
            "pdflatex is not installed."
        )

    print("\nResumeForge complete.")


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
        required=True,
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