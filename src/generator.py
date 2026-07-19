import sys
import os

# Add root directory to path to import local_llm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import local_llm

class ResumeGenerator:
    def __init__(self, model_name: str = "qwen-7b"):
        # We default to gemma-2b since it's the default in local_llm.py, 
        # but the user can pass larger models.
        self.llm = local_llm.load(model_name)
        
    def generate_resume(self, jd_text: str, retrieved_chunks: list[str]) -> str:
        prompt = f"""
You are an expert resume writer. 
Your task is to generate a highly tailored, ATS-friendly resume summary and experience bullet points based on the provided Job Description.

CRITICAL RULE: You MUST ONLY use the facts, skills, and experiences present in the 'Source Resume Chunks'. 
DO NOT invent, fabricate, or hallucinate any skills, metrics, or experiences that are not explicitly stated in the source chunks.

Job Description:
{jd_text}

Source Resume Chunks (Verifiable Facts):
{chr(10).join(retrieved_chunks)}

Generate the tailored resume content in Markdown format.
"""
        messages = [{"role": "user", "content": prompt.strip()}]
        return self.llm.generate(messages, local_llm.GenConfig(max_new_tokens=1024, temperature=0.1))

    def rewrite_resume(self, draft: str, critique: str, retrieved_chunks: list[str]) -> str:
        prompt = f"""
You are an expert resume writer. 
You previously wrote a draft resume, but it contains hallucinations (claims not supported by the source material).

Your task is to REWRITE the draft to fix the issues mentioned in the Critique.
Remove or rephrase any unsupported claims. 

Source Resume Chunks (Verifiable Facts):
{chr(10).join(retrieved_chunks)}

Draft Resume:
{draft}

Critique (Hallucinations to fix):
{critique}

Provide the rewritten, corrected resume in Markdown format.
"""
        messages = [{"role": "user", "content": prompt.strip()}]
        return self.llm.generate(messages, local_llm.GenConfig(max_new_tokens=1024, temperature=0.1))
