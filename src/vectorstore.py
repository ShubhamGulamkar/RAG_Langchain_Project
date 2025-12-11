import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        
        # Chunk the documents
        chunks = emb_pipe.chunk_documents(documents)
        
        # Get the embeddings for those chunks
        embeddings = emb_pipe.embed_chunks(chunks)
        
        # Debug: Ensure embeddings are correctly generated
        print(f"[INFO] Embeddings generated for {len(chunks)} chunks with shape: {embeddings.shape}")
        
        # Ensure the embeddings are in the correct format (float32)
        embeddings = embeddings.astype(np.float32)
        
        # Prepare metadata for the chunks
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        
        # Add the embeddings to the Faiss index
        self.add_embeddings(embeddings, metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        
        # If the Faiss index doesn't exist, create it
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)  # Using L2 distance metric
            print(f"[INFO] Created new Faiss index with dimension: {dim}")
        
        # Debug: Print the first few embeddings to ensure they are correct
        print(f"[DEBUG] First 5 embeddings (dim {dim}): {embeddings[:5]}")

        # Add embeddings to the Faiss index
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        """Save the Faiss index and metadata to disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        # Save the Faiss index
        faiss.write_index(self.index, faiss_path)
        
        # Save the metadata
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        """Load the Faiss index and metadata from disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        # Check if Faiss index exists
        if not os.path.exists(faiss_path):
            print(f"[INFO] Faiss index not found at {faiss_path}. Please build the index first.")
            return
        
        # Load the Faiss index
        self.index = faiss.read_index(faiss_path)
        
        # Load metadata
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for the closest vectors to the query embedding in the Faiss index."""
        D, I = self.index.search(query_embedding, top_k)  # Perform search
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        """Perform a query and retrieve the closest vectors from the Faiss index."""
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype(np.float32)  # Embed query text
        
        # Debug: Ensure query embedding is correctly created
        print(f"[INFO] Query embedding shape: {query_emb.shape}")
        
        return self.search(query_emb, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    
    # Create the Faiss vector store
    store = FaissVectorStore("faiss_store")
    
    # Build the vector store and save the Faiss index
    store.build_from_documents(docs)
    
    # Load the Faiss index from the saved file
    store.load()
    
    # Query the vector store
    result = store.query("what is Modular RAG?", top_k=3)
    print(result)

