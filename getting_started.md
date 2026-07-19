# Getting Started with Self-Healing Resume RAG

Welcome to the Self-Healing Resume RAG system! This tool uses local LLMs to generate highly tailored, ATS-friendly resumes based on a specific Job Description. Most importantly, it features an **Anti-Hallucination Evaluator** loop to ensure it never invents skills or experiences you don't actually have.

## 1. Prerequisites

Ensure you have installed the dependencies using `uv` as described in the `setup.md`. 
If you haven't yet, run:
```bash
uv sync --inexact
```

## 2. Activate the Environment

Before running any scripts, you must activate the virtual environment:

**Linux & macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
```

## 3. Prepare Your Data

The system needs two inputs:
1. **Master Resume:** This is a comprehensive list of all your work history, skills, and projects. It serves as the "Ground Truth". Supported formats: `.pdf`, `.md`, or `.txt`.
2. **Job Description (JD):** A text file containing the job description you are applying for.

*Note: We have provided `sample_resume.md` and `sample_jd.txt` in the root folder for you to test with.*

## 4. Run the Pipeline

To generate a resume, pass your master resume and the job description to the main script:

```bash
python src/main.py path/to/master_resume.pdf path/to/job_description.txt
```

**Example Test Run:**
```bash
python src/main.py sample_resume.md sample_jd.txt
```

## 5. What to Expect (The Output)

When you run the command, you will see the pipeline execute in real-time in your terminal:

1. **Ingestion & DB Setup:** It will parse your master resume, chunk it, and load it into a local ChromaDB vector database.
2. **Retrieval:** It will scan the JD and retrieve only the most relevant experiences from your master resume.
3. **Drafting:** The local LLM (`qwen-7b` by default) will draft an initial resume.
4. **Self-Healing / Anti-Hallucination Loop:** 
   - A strict "Evaluator" prompt will cross-reference the generated draft against your actual retrieved facts.
   - If it detects a hallucinated skill (e.g., claiming you know Rust when your resume only mentions Python), it will reject the draft and force a rewrite.
   - It will iterate until the resume is 100% factually accurate.
5. **Final Output:** The final, verified resume will be saved to your root directory as `final_resume.md`.

## 6. Advanced Usage: Changing Models

By default, the script uses `qwen-7b` (a powerful, open model). If you wish to use a different model (e.g., `gemma-2b`, which is faster but requires Hugging Face authentication), you can modify the `model_name` variable in `src/main.py` or pass it as an argument if you extend the CLI. 

See `local_llm.py` for all available model profiles!
