import sys
import os

from ingest import process_resume
from rag import ResumeRAG
from generator import ResumeGenerator
from evaluator import HallucinationEvaluator

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <path_to_master_resume> <path_to_jd>")
        sys.exit(1)
        
    master_resume_path = sys.argv[1]
    jd_path = sys.argv[2]
    
    # Use qwen-7b by default as it is ungated
    model_name = "qwen-7b"
    
    # 1. Ingest Master Resume
    print("--- 1. Ingesting Master Resume ---")
    chunks = process_resume(master_resume_path)
    print(f"Extracted {len(chunks)} chunks.")
    
    # 2. Setup RAG
    print("--- 2. Setting up Vector DB ---")
    rag = ResumeRAG()
    rag.add_chunks(chunks)
    
    # Read JD
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_text = f.read()
        
    # 3. Retrieve Relevant Context
    print("--- 3. Retrieving Relevant Experience ---")
    retrieved_chunks = rag.retrieve(jd_text, n_results=5)
    print(f"Retrieved {len(retrieved_chunks)} chunks for context.")
    
    # 4. Generate Draft
    print("--- 4. Generating Initial Draft ---")
    generator = ResumeGenerator(model_name=model_name) 
    draft = generator.generate_resume(jd_text, retrieved_chunks)
    print("\n=== INITIAL DRAFT ===")
    print(draft)
    
    # 5. Evaluate and Heal Loop
    print("\n--- 5. Self-Healing Anti-Hallucination Loop ---")
    evaluator = HallucinationEvaluator(model_name=model_name)
    
    max_iterations = 3
    for i in range(max_iterations):
        print(f"\n[Iteration {i+1}] Evaluating draft...")
        eval_result = evaluator.evaluate(draft, retrieved_chunks)
        
        has_hallucinations = eval_result.get("has_hallucinations", False)
        critique = eval_result.get("critique", "None")
        
        if not has_hallucinations:
            print("✅ Evaluation Passed! No hallucinations detected.")
            break
        else:
            print(f"❌ Hallucinations detected:\n{critique}")
            print("🔄 Rewriting to fix hallucinations...")
            draft = generator.rewrite_resume(draft, critique, retrieved_chunks)
            print("\n=== REVISED DRAFT ===")
            print(draft)
            
    if has_hallucinations:
        print("\n⚠️ Warning: Reached max iterations, but minor hallucinations may still exist.")
    
    print("\n=== FINAL APPROVED RESUME ===")
    print(draft)
    
    # Output to file
    with open("final_resume.md", "w") as f:
        f.write(draft)
    print("\nSaved to final_resume.md")

if __name__ == "__main__":
    main()
