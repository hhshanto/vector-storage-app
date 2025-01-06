import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.storage.base import VectorStore

class FAISSStore(VectorStore):
    def __init__(self, dimension: int, index_type: str = "L2"):
        """
        Initialize FAISS vector store.
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize the FAISS index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Dictionary to store metadata
        self.metadata_dict: Dict[str, Dict[str, Any]] = {}
        
        # Dictionary to map FAISS internal IDs to user-provided IDs
        self.id_map: Dict[int, str] = {}

    def add_vectors(self, 
                   vectors: np.ndarray, 
                   ids: List[str], 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add vectors to the store with their IDs and optional metadata.
        """
        # Input validation
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}")
        
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        
        if metadata is not None and len(metadata) != len(vectors):
            raise ValueError("Number of metadata entries must match number of vectors")
        
        # Add vectors to FAISS index
        start_idx = self.index.ntotal
        self.index.add(vectors.astype(np.float32))
        
        # Store metadata and map IDs
        for i, id_str in enumerate(ids):
            faiss_id = start_idx + i
            self.id_map[faiss_id] = id_str
            
            if metadata is not None:
                self.metadata_dict[id_str] = metadata[i]

    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors and return IDs, distances, and metadata.
        """
        # Ensure query vector has correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Perform search
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for invalid results
                id_str = self.id_map[idx]
                metadata = self.metadata_dict.get(id_str, {})
                results.append((id_str, float(dist), metadata))
        
        return results

    def save(self, file_path: str) -> None:
        """
        Save the vector store to disk.
        """
        # Save FAISS index
        faiss.write_index(self.index, f"{file_path}.index")
        
        # Save metadata and ID mappings
        np.savez(f"{file_path}.meta",
                 id_map=np.array(list(self.id_map.items())),
                 metadata=np.array(list(self.metadata_dict.items()), dtype=object))

    def load(self, file_path: str) -> None:
        """
        Load the vector store from disk.
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{file_path}.index")
        
        # Load metadata and ID mappings
        data = np.load(f"{file_path}.meta", allow_pickle=True)
        self.id_map = dict(data['id_map'])
        self.metadata_dict = dict(data['metadata'])