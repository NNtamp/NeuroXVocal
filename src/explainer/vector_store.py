import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

"""
This module handles the creation and management of vector embeddings
for literature and patient data using FAISS.
"""


class VectorStore:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.literature_index = None
        self.literature_texts = []
        
    def create_literature_index(self, literature: List[str]):
        """Create FAISS index for literature chunks."""
        chunks = []
        for doc in literature:
            paragraphs = doc.split('\n\n')
            for para in paragraphs:
                sentences = para.split('. ')
                chunks.extend(sentences)
        
        self.literature_texts = chunks
        embeddings = self.encoder.encode(chunks)
        
        dimension = embeddings.shape[1]
        self.literature_index = faiss.IndexFlatL2(dimension)
        self.literature_index.add(embeddings.astype('float32'))
        
    def get_relevant_literature(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant literature chunks for a query."""
        query_embedding = self.encoder.encode([query])
        D, I = self.literature_index.search(query_embedding.astype('float32'), k)
        return [self.literature_texts[i] for i in I[0]]
