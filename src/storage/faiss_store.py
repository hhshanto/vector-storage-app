import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from .base import VectorStore

class FAISSStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, dimension: int, index_type: str = "L2"):
        """
        Initialize FAISS store.
        
        Args:
            dimension (int): Dimensionality of the vectors
            index_type (str): Type of FAISS index ("L2" or "IP" for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store for IDs and metadata
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # Store original vectors (needed for get_vector method)
        self.vectors: List[np.ndarray] = []
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add vectors to the FAISS index.
        
        Args:
            vectors (np.ndarray): Vectors to add (shape: n_vectors x dimension)
            ids (List[str]): List of IDs corresponding to the vectors
            metadata (Optional[List[Dict[str, Any]]]): Optional metadata for each vector
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
        
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]
        
        # Add vectors to FAISS index
        self.index.add(vectors.astype(np.float32))
        
        # Store vectors, IDs and metadata
        self.vectors.extend([v for v in vectors])
        self.ids.extend(ids)
        self.metadata.extend(metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector (np.ndarray): Query vector
            k (int): Number of results to return
        
        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: List of (id, distance, metadata) tuples
        """
        # Ensure query vector has correct shape
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Perform search
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS returns -1 for invalid results
                results.append((
                    self.ids[idx],
                    float(distance),
                    self.metadata[idx]
                ))
        
        return results
    
    def get_vector(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a specific vector by its ID.
        
        Args:
            id (str): ID of the vector to retrieve
        
        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: Tuple of (vector, metadata) if found,
                                                        None if not found
        """
        try:
            idx = self.ids.index(id)
            return self.vectors[idx], self.metadata[idx]
        except ValueError:
            return None
    
    def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors from the store by their IDs.
        Note: FAISS doesn't support direct deletion, so we need to rebuild the index
        
        Args:
            ids (List[str]): List of IDs to delete
        """
        # Get indices to keep
        indices_to_delete = set(i for i, id in enumerate(self.ids) if id in ids)
        indices_to_keep = [i for i in range(len(self.ids)) if i not in indices_to_delete]
        
        if not indices_to_keep:
            self.clear()
            return
        
        # Create new arrays with kept items
        kept_vectors = np.array([self.vectors[i] for i in indices_to_keep])
        kept_ids = [self.ids[i] for i in indices_to_keep]
        kept_metadata = [self.metadata[i] for i in indices_to_keep]
        
        # Clear current data
        self.clear()
        
        # Add kept vectors back
        self.add_vectors(kept_vectors, kept_ids, kept_metadata)
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        # Reset FAISS index
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:  # IP
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Clear stored data
        self.ids.clear()
        self.metadata.clear()
        self.vectors.clear()
    
    def save(self, file_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            file_path (str): Path to save the store
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{file_path}.index")
        
        # Save vectors, IDs and metadata
        with open(f"{file_path}.meta", 'w') as f:
            json.dump({
                'ids': self.ids,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'vectors': [v.tolist() for v in self.vectors]
            }, f)
    
    def load(self, file_path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            file_path (str): Path to load the store from
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{file_path}.index")
        
        # Load metadata and vectors
        with open(f"{file_path}.meta", 'r') as f:
            data = json.load(f)
            self.ids = data['ids']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.vectors = [np.array(v) for v in data['vectors']]
    
    @classmethod
    def from_embeddings(cls, embeddings_file: str, index_type: str = "L2") -> 'FAISSStore':
        """
        Create a FAISSStore instance from a saved embeddings file.
        
        Args:
            embeddings_file (str): Path to the .npz embeddings file
            index_type (str): Type of FAISS index to create ("L2" or "IP")
        
        Returns:
            FAISSStore: A new instance with the loaded embeddings
        """
        with np.load(embeddings_file, allow_pickle=True) as data:
            # Get model info
            model_info = json.loads(str(data['__model_info__']))
            dimension = model_info['embedding_dim']
            
            # Create store
            store = cls(dimension=dimension, index_type=index_type)
            
            # Get embeddings and IDs
            ids = [key for key in data.files if key != '__model_info__']
            vectors = np.array([data[id] for id in ids])
            
            # Add to store with model info in metadata
            store.add_vectors(
                vectors=vectors,
                ids=ids,
                metadata=[{'model_info': model_info} for _ in ids]
            )
            
            return store
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Dictionary containing store statistics
        """
        return {
            'num_vectors': len(self.ids),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'memory_usage': self.index.ntotal * self.dimension * 4  # 4 bytes per float32
        }
    
    def batch_search(self, 
                    query_vectors: np.ndarray, 
                    k: int = 5) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Search for multiple query vectors at once.
        
        Args:
            query_vectors (np.ndarray): Vectors to search for. Shape should be (n_queries, dimension)
            k (int): Number of nearest neighbors to return for each query
        
        Returns:
            List[List[Tuple[str, float, Dict[str, Any]]]]: List of search results for each query
        """
        # Ensure query vectors have correct shape
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        # Perform batch search
        distances, indices = self.index.search(query_vectors.astype(np.float32), k)
        
        # Format results
        results = []
        for query_indices, query_distances in zip(indices, distances):
            query_results = []
            for idx, distance in zip(query_indices, query_distances):
                if idx != -1:  # FAISS returns -1 for invalid results
                    query_results.append((
                        self.ids[idx],
                        float(distance),
                        self.metadata[idx]
                    ))
            results.append(query_results)
        
        return results
