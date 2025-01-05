from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class VectorStore(ABC):
    """
    Abstract base class for vector storage implementations.
    
    This class defines the interface that all vector store implementations must follow.
    It provides a standard set of methods for storing, retrieving, and searching vectors,
    along with their associated metadata.
    
    Methods that must be implemented by subclasses:
        - add_vectors: Add vectors to the store
        - search: Search for similar vectors
        - save: Save the vector store to disk
        - load: Load the vector store from disk
    """
    
    @abstractmethod
    def add_vectors(self, 
                   vectors: np.ndarray, 
                   ids: List[str], 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add vectors to the store with their IDs and optional metadata.
        
        Args:
            vectors (np.ndarray): Array of vectors to store. Shape should be (n_vectors, dimension)
            ids (List[str]): List of unique identifiers for each vector
            metadata (Optional[List[Dict[str, Any]]]): Optional metadata for each vector
                Each metadata dict can contain arbitrary information about the vector
        
        Raises:
            ValueError: If lengths of vectors, ids, and metadata don't match
            ValueError: If vector dimensions don't match the store's configuration
        """
        pass
    
    @abstractmethod
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors and return IDs, distances, and metadata.
        
        Args:
            query_vector (np.ndarray): Vector to search for. Shape should be (dimension,)
            k (int): Number of nearest neighbors to return
        
        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: List of tuples containing:
                - str: ID of the similar vector
                - float: Distance/similarity score
                - Dict[str, Any]: Metadata associated with the vector
        
        Raises:
            ValueError: If query_vector dimensions don't match the store's configuration
        """
        pass
    
    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            file_path (str): Path where the vector store should be saved
        
        Raises:
            IOError: If the file cannot be written
            Exception: If the store's state cannot be serialized
        """
        pass
    
    @abstractmethod
    def load(self, file_path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            file_path (str): Path from where the vector store should be loaded
        
        Raises:
            IOError: If the file cannot be read
            ValueError: If the file format is invalid
            Exception: If the store's state cannot be deserialized
        """
        pass

    def batch_search(self, 
                    query_vectors: np.ndarray, 
                    k: int = 5) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Search for multiple query vectors at once.
        
        This is a default implementation that can be overridden by subclasses
        for more efficient batch processing.
        
        Args:
            query_vectors (np.ndarray): Vectors to search for. Shape should be (n_queries, dimension)
            k (int): Number of nearest neighbors to return for each query
        
        Returns:
            List[List[Tuple[str, float, Dict[str, Any]]]]: List of search results for each query
        """
        return [self.search(query_vector, k) for query_vector in query_vectors]

    def get_vector(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a specific vector by its ID.
        
        This is a default implementation that can be overridden by subclasses
        for more efficient retrieval.
        
        Args:
            id (str): ID of the vector to retrieve
        
        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: Tuple of (vector, metadata) if found,
                                                        None if not found
        """
        raise NotImplementedError("get_vector method not implemented")

    def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors from the store by their IDs.
        
        This is a default implementation that can be overridden by subclasses
        for more efficient deletion.
        
        Args:
            ids (List[str]): List of IDs to delete
        
        Raises:
            KeyError: If any of the IDs don't exist in the store
        """
        raise NotImplementedError("delete_vectors method not implemented")

    def clear(self) -> None:
        """
        Clear all vectors from the store.
        
        This is a default implementation that can be overridden by subclasses
        for more efficient clearing.
        """
        raise NotImplementedError("clear method not implemented")