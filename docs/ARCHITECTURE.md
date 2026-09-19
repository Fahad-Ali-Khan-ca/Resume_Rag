# Architecture

Resume RAG is designed as an evidence-grounded multi-stage generation system.

The system separates factual extraction from final resume writing so that generated claims can be traced back to source documents.

## High-Level Architecture

```text
┌───────────────────────────────┐
│      Candidate Documents      │
│   PDF / Markdown / Text       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│            Parser             │
│                               │
│ Converts source files into    │
│ normalized document objects   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│            Chunker            │
│                               │
│ Breaks documents into smaller │
│ evidence-sized text segments  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Evidence Extractor       │
│                               │
│ Uses the LLM to convert raw   │
│ chunks into structured facts  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Structured Evidence      │
│                               │
│ skills / projects / impact /  │
│ technologies / responsibilities│
└───────────────┬───────────────┘
                │
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
┌──────────────┐   ┌──────────────┐
│ Job          │   │ Candidate    │
│ Description  │   │ Evidence     │
└──────┬───────┘   └──────┬───────┘
       │                  │
       └────────┬─────────┘
                ▼
┌───────────────────────────────┐
│ Retrieval / Evidence Ranking  │
│                               │
│ Selects evidence most useful  │
│ for the target role           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Resume Generation        │
│                               │
│ Writes tailored content using │
│ only supported evidence       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       Claim Validation        │
│                               │
│ Checks that generated claims  │
│ remain grounded in evidence   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        LaTeX Renderer         │
│                               │
│ Converts structured resume    │
│ content into final document   │
└───────────────────────────────┘
```

## Main Components

### `ingestion/parser.py`

Responsibility:

```text
Raw file
   │
   ▼
Normalized document text + metadata
```

The parser isolates file-format-specific logic from the rest of the system.

This lets downstream components work with one normalized representation regardless of whether the source was PDF, Markdown, or text.

### `ingestion/chunker.py`

Responsibility:

```text
Document
   │
   ▼
Chunk 1
Chunk 2
Chunk 3
...
```

The chunker limits the amount of text processed in a single extraction step and creates smaller evidence units that are easier to retrieve later.

Each chunk should retain enough metadata to trace it back to its source document.

### `ingestion/evidence_extractor.py`

Responsibility:

```text
Text chunk
    │
    ▼
LLM extraction
    │
    ▼
Structured evidence
```

Instead of storing only raw text embeddings, this stage attempts to identify explicit candidate facts.

Examples include:

- technologies used
- responsibilities
- projects
- measurable outcomes
- software systems
- programming languages
- tools
- domain experience

The extractor should preserve source traceability.

### `llm.py`

Defines the common interface used by language-model implementations.

Conceptually:

```python
class LLM:
    def generate(...):
        ...

    def stream(...):
        ...
```

The rest of the pipeline should depend on this abstraction rather than directly depending on Hugging Face.

### `local_llm.py`

Implements the LLM interface using a locally hosted model.

Current development has used Hugging Face Transformers and Gemma-family models.

This layer is responsible for concerns such as:

- model loading
- tokenizer loading
- device selection
- dtype
- generation parameters
- prompt execution

### Retrieval / Ranking

This stage compares the target job description with candidate evidence.

Its purpose is not simply semantic similarity.

The long-term objective is to select evidence that is:

```text
Relevant
+
Specific
+
Defensible
+
Useful for the target role
```

### Resume Generation

The generation stage receives selected evidence rather than the entire raw corpus.

This reduces irrelevant context and makes grounding easier.

Conceptually:

```text
Job requirements
      +
Selected evidence
      +
Resume formatting instructions
      │
      ▼
Generated resume content
```

### Claim Validation

Before a generated claim is accepted, the system should verify that it is supported by candidate evidence.

Conceptually:

```text
Generated claim
      │
      ▼
Find supporting evidence
      │
      ├── evidence exists ──► keep / refine
      │
      └── unsupported ──────► reject
```

This stage is important for autonomous operation.

### `rendering/latex_renderer.py`

The renderer converts structured resume data into the final LaTeX representation.

Rendering is kept separate from content generation so that resume formatting can evolve independently from the LLM pipeline.

## Design Principle

The primary architectural rule is:

```text
Raw candidate data
        ↓
structured evidence
        ↓
relevant evidence
        ↓
generated claim
        ↓
validated claim
        ↓
resume
```

The LLM should not be treated as the source of truth.

The candidate corpus is the source of truth.

## Relationship to Autonomous Job Applier

Resume RAG is one module in the larger architecture:

```text
                   Autonomous Job Applier

┌─────────────────┐
│   Job Scrapers  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Job Normalizer  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Resume RAG    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Application     │
│ Package         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Job Applier   │
└─────────────────┘
```

The resume system should eventually accept a normalized job posting and return a job-specific application-ready resume without manual rewriting.
