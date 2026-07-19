import chromadb
from chromadb.utils import embedding_functions
from typing import List

class ResumeRAG:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "resume_chunks"):
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use sentence-transformers for local embeddings
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.ef
        )

    def add_chunks(self, chunks: List[str]):
        """Adds text chunks to the vector database."""
        if not chunks:
            return
            
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        # Clear existing for simplicity if this is a single-resume system, 
        # but let's just add for now (we might get duplicates if run multiple times).
        # A better approach is to clear or hash the chunks.
        
        # Let's clear and re-add for simplicity of the workflow
        try:
            existing = self.collection.get()
            if existing['ids']:
                self.collection.delete(ids=existing['ids'])
        except Exception:
            pass

        self.collection.add(
            documents=chunks,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks to Vector DB.")

    def retrieve(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieves the most relevant chunks for a given query (Job Description)."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if results and results['documents']:
            return results['documents'][0]
        return []

if __name__ == "__main__":
    # Test
    rag = ResumeRAG()
    rag.add_chunks(["I am a software engineer with 5 years of Python experience.", 
                    "I built a scalable RAG system using ChromaDB.",
                    "I like playing guitar."])
    
    print(rag.retrieve("Looking for a Python developer with RAG experience", n_results=2))
