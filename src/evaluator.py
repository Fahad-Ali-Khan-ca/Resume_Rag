import sys
import os
import json

# Add root directory to path to import local_llm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import local_llm

class HallucinationEvaluator:
    def __init__(self, model_name: str = "qwen-7b"):
        # For evaluation, a smarter model is usually better. 
        self.llm = local_llm.load(model_name)
        
    def evaluate(self, generated_text: str, source_chunks: list[str]) -> dict:
        """
        Evaluates if the generated text contains hallucinations not present in the source chunks.
        Returns a dict with 'has_hallucinations' (bool) and 'critique' (str).
        """
        prompt = f"""
You are a strict factual consistency evaluator.
Your task is to verify if the claims made in the Generated Resume are fully supported by the Source Chunks.

Source Chunks (The absolute ground truth):
{chr(10).join(source_chunks)}

Generated Resume:
{generated_text}

Evaluate the Generated Resume. Are there any skills, metrics, job titles, or experiences mentioned that do NOT exist in the Source Chunks?

Return your analysis in the following JSON format ONLY:
{{
    "has_hallucinations": true or false,
    "critique": "If true, list the specific unsupported claims here. If false, output 'None'."
}}
"""
        messages = [{"role": "user", "content": prompt.strip()}]
        # Lower temperature for strict evaluation
        response = self.llm.generate(messages, local_llm.GenConfig(max_new_tokens=512, temperature=0.0))
        
        try:
            # Basic cleanup if the LLM wrapped it in markdown code blocks
            clean_json = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            return result
        except Exception as e:
            # Fallback if parsing fails - assume hallucination to be safe to trigger a rewrite
            print(f"[Evaluator Parse Error]: {e}\nRaw output: {response}")
            return {
                "has_hallucinations": True, 
                "critique": "Evaluator could not parse the output. Please ensure you strictly follow the source text."
            }
