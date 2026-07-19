import re
from typing import List
from pypdf import PdfReader

def parse_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, max_chunk_size: int = 800) -> List[str]:
    """
    Chunks text by paragraphs or bullet points to keep related context together.
    """
    # Split by double newline to get rough paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # If adding this paragraph exceeds max size, save current chunk and start a new one
        if len(current_chunk) + len(p) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            
        # If a single paragraph is longer than max_chunk_size, we just add it anyway 
        # (could do finer sentence chunking, but for resumes this is rare)
        current_chunk += p + "\n\n"
        
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def process_resume(file_path: str) -> List[str]:
    """Parses a resume and returns a list of chunks."""
    if file_path.endswith('.pdf'):
        text = parse_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    return chunk_text(text)

if __name__ == "__main__":
    # Test logic if run directly
    import sys
    if len(sys.argv) > 1:
        chunks = process_resume(sys.argv[1])
        for i, c in enumerate(chunks):
            print(f"--- Chunk {i+1} ---")
            print(c)
