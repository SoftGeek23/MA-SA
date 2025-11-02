"""FAISS-based episodic memory for state+reflection retrieval."""
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pickle
import json

from sentence_transformers import SentenceTransformer


class EpisodicMemory:
    """FAISS-based episodic memory for storing and retrieving state+reflection pairs."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_dim: int = 384,
        k_neighbors: int = 5,
        index_path: Optional[str] = None
    ):
        """Initialize episodic memory.
        
        Args:
            embedding_model: Name of sentence-transformers model
            index_dim: Dimension of embeddings
            k_neighbors: Default number of neighbors for kNN search
            index_path: Path to save/load FAISS index
        """
        self.embedding_model_name = embedding_model
        self.index_dim = index_dim
        self.k_neighbors = k_neighbors
        self.index_path = Path(index_path) if index_path else None
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(index_dim)
        
        # Store metadata for each vector (state + reflection)
        self.metadata: List[Dict[str, Any]] = []
        
        # Load existing index if path exists
        if self.index_path and self.index_path.exists():
            self.load()
    
    def encode_state_and_reflection(self, state_text: str, reflection: str) -> np.ndarray:
        """Encode state + reflection into embedding.
        
        Args:
            state_text: State representation as text
            reflection: Reflection text (rules + antirules)
            
        Returns:
            Embedding vector
        """
        # Concatenate state and reflection
        combined_text = f"State: {state_text}\nReflection: {reflection}"
        
        # Get embedding
        embedding = self.embedder.encode(combined_text, convert_to_numpy=True)
        
        # Ensure correct dimension
        if embedding.shape[0] != self.index_dim:
            # Handle dimension mismatch (shouldn't happen with correct model)
            if embedding.shape[0] > self.index_dim:
                embedding = embedding[:self.index_dim]
            else:
                embedding = np.pad(embedding, (0, self.index_dim - embedding.shape[0]))
        
        return embedding.reshape(1, -1)
    
    def add_memory(
        self,
        state_text: str,
        reflection: str,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a state+reflection pair to memory.
        
        Args:
            state_text: State representation as text
            reflection: Reflection text
            episode_id: Optional episode ID
            task_id: Optional task ID
            metadata: Optional additional metadata
        """
        # Encode state + reflection
        embedding = self.encode_state_and_reflection(state_text, reflection)
        
        # Add to FAISS index
        self.index.add(embedding.astype('float32'))
        
        # Store metadata
        memory_entry = {
            "episode_id": episode_id,
            "task_id": task_id,
            "state_text": state_text,
            "reflection": reflection,
            "metadata": metadata or {}
        }
        self.metadata.append(memory_entry)
    
    def search(
        self,
        state_text: str,
        k: Optional[int] = None,
        reflection_hint: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar states with reflections.
        
        Args:
            state_text: Current state representation
            k: Number of neighbors (defaults to self.k_neighbors)
            reflection_hint: Optional reflection text to include in query
            
        Returns:
            List of (distance, memory_entry) tuples, sorted by distance
        """
        if self.index.ntotal == 0:
            return []
        
        k = k or self.k_neighbors
        k = min(k, self.index.ntotal)  # Can't retrieve more than total
        
        # Encode query (with or without reflection hint)
        if reflection_hint:
            query_embedding = self.encode_state_and_reflection(state_text, reflection_hint)
        else:
            # Just encode state if no reflection hint
            query_text = f"State: {state_text}"
            query_embedding = self.embedder.encode(query_text, convert_to_numpy=True)
            if query_embedding.shape[0] != self.index_dim:
                if query_embedding.shape[0] > self.index_dim:
                    query_embedding = query_embedding[:self.index_dim]
                else:
                    query_embedding = np.pad(
                        query_embedding, (0, self.index_dim - query_embedding.shape[0])
                    )
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve metadata
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                results.append((float(dist), self.metadata[idx]))
        
        return results
    
    def get_nearest_reflections(self, state_text: str, k: Optional[int] = None) -> List[str]:
        """Get just the reflection texts from nearest neighbors.
        
        Args:
            state_text: Current state representation
            k: Number of neighbors
            
        Returns:
            List of reflection texts
        """
        results = self.search(state_text, k=k)
        return [reflection for _, entry in results if "reflection" in entry]
    
    def save(self, path: Optional[str] = None):
        """Save FAISS index and metadata to disk.
        
        Args:
            path: Optional override path
        """
        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise ValueError("No index path specified for saving")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path.with_suffix('.index')
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata_file = save_path.with_suffix('.metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config_file = save_path.with_suffix('.config.json')
        config = {
            "embedding_model": self.embedding_model_name,
            "index_dim": self.index_dim,
            "k_neighbors": self.k_neighbors,
            "num_vectors": self.index.ntotal
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: Optional[str] = None):
        """Load FAISS index and metadata from disk.
        
        Args:
            path: Optional override path
        """
        load_path = Path(path) if path else self.index_path
        if load_path is None:
            raise ValueError("No index path specified for loading")
        
        if not load_path.exists():
            return  # Nothing to load
        
        # Load FAISS index
        index_file = load_path.with_suffix('.index')
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
        
        # Load metadata
        metadata_file = load_path.with_suffix('.metadata.pkl')
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        
        # Verify dimensions match
        if self.index.ntotal > 0:
            assert len(self.metadata) == self.index.ntotal, \
                "Index and metadata size mismatch"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "num_memories": self.index.ntotal,
            "index_dim": self.index_dim,
            "embedding_model": self.embedding_model_name,
            "k_neighbors": self.k_neighbors
        }

