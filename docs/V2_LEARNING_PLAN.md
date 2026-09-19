# ResumeForge v2 — code ownership and defense plan

Branch: `v2/development`. Baseline: v1 one-page implementation on main after documentation update. Do not rewrite or move the `v1` tag.

## Working method for every module

1. State the module's one-sentence responsibility.
2. Explain each import, type, class, function, branch, exception, and I/O side effect.
3. Trace a real source record and every field it carries from input to output.
4. State the module's invariants (what must always remain true) and trust boundaries.
5. Write a small failing test for a concrete flaw, then implement the smallest fix.
6. Run unit tests, offline integration smoke tests, and a real model/PDF run when affected.
7. Record trade-offs and evidence of test results in the pull request/commit.

A model-written explanation is not a substitute for reproducing the behavior locally, predicting outputs, and defending the design without the model.

## Review order

1. `config.py`, `run_pipeline.py`: orchestration, paths, function signatures, exceptions, intermediate artifacts.
2. `ingestion/canonical_parser.py`, `parser.py`, `chunker.py`, `candidate_profile.py`: entity semantics, text normalization, metadata, stable IDs.
3. `ingestion/evidence_extractor.py`: Pydantic schemas, prompts, caching, deterministic hash, metric support, duplicate claims.
4. `retrieval/common.py`, `bm25.py`, `semantic.py`, `hybrid.py`, `reranker.py`: indexing, BM25/embedding math, rank fusion, rerank and evaluation.
5. `jd/schemas.py`, `requirement_extractor.py`, `matching/matcher.py`, `evidence_map.py`: requirement truth conditions, numeric thresholds, match confidence vs eligibility.
6. `generation/planner.py`, `bullet_generator.py`, `resume_generator.py`, `composition.py`, `validation/claim_validator.py`: evidence ownership, deduplication, factual validation.
7. `rendering/latex_renderer.py`, `page_fit.py`: escaping, page geometry, overflow/underfill policy and PDF measurement.
8. `local_llm.py`, `pyproject.toml`, tests, packaging: model abstraction, GPU/CPU choice, quantization, resource limits, reproducible runs.

## First v2 change — requirement truth gate

Problem observed in v1: an LLM can output STRONG for “2–3 years of experience” while only short, dated internships are present. The current evidence mapper accepts the model's `match_strength` and selected IDs without independently checking a duration threshold.

Implement a **pure, deterministic check** for explicitly stated minimum experience only. Begin with complete `YYYY-MM` employment intervals; calculate the union of calendar months, not sum of overlapping jobs. If **all** relevant entries' date ranges are complete and the upper bound on recorded duration is below the required minimum, a model's STRONG rating must be capped. An incomplete/undated entry makes the duration unknown, not automatically zero. A sufficient duration must never automatically promote a result to STRONG: role/domain relevance and actual hands-on work still require evidence.

Before integration, write unit tests for: two four-month internships vs 2 years; overlapping intervals; a 24-month span; missing or ongoing dates; a requirement without a numeric threshold; and a 5-year specialized-domain requirement where generic 5-year work should not automatically pass.

Integration path: `run_pipeline.py` supplies candidate structure to a post-matching guard, or a dedicated matching API accepts the profile. Record whether the gate changed a match and why. Do not change `years_experience` to a misleading boolean. Keep v1 model outputs as comparison fixtures.

## Release criteria

- No cross-entity evidence attribution or unverified numeric claims.
- Quantitative JD requirements do not receive unjustified strong matches.
- Skills are explicit technical capabilities, not abstract activities or project features.
- Distinct accomplishment selection does not repeat one achievement as multiple bullets.
- One-page rendering does not invent content or silently discard the only supported bullet for a career entry.
- Tests include adversarial inputs and model-independent fixtures; report real model behavior separately.
- Each change can be explained from inputs, through syntax/control flow, to observable output.
