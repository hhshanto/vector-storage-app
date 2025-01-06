# src/storage/utils/validation.py
import numpy as np
from typing import List, Dict, Any, Optional

def validate_vectors(vectors: np.ndarray, dimension: int) -> None:
    """
    Validate vector array dimensions and type.
    
    Args:
        vectors (np.ndarray): Array of vectors to validate
        dimension (int): Expected dimension of vectors
    
    Raises:
        ValueError: If vectors don't match expected format
        TypeError: If vectors aren't the correct type
    """
    if not isinstance(vectors, np.ndarray):
        raise TypeError("Vectors must be a numpy array")
    
    if len(vectors.shape) != 2:
        raise ValueError("Vectors must be a 2D array")
    
    if vectors.shape[1] != dimension:
        raise ValueError(f"Expected vectors of dimension {dimension}, got {vectors.shape[1]}")

def validate_ids(ids: List[str], num_vectors: int) -> None:
    """
    Validate vector IDs.
    
    Args:
        ids (List[str]): List of IDs to validate
        num_vectors (int): Expected number of IDs
    
    Raises:
        ValueError: If IDs don't match expected format
    """
    if len(ids) != num_vectors:
        raise ValueError(f"Number of IDs ({len(ids)}) doesn't match number of vectors ({num_vectors})")
    
    if len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique")

def validate_metadata(metadata: Optional[List[Dict[str, Any]]], num_vectors: int) -> None:
    """
    Validate metadata format and length.
    
    Args:
        metadata (Optional[List[Dict[str, Any]]]): Metadata to validate
        num_vectors (int): Expected number of metadata entries
    
    Raises:
        ValueError: If metadata doesn't match expected format
    """
    if metadata is not None and len(metadata) != num_vectors:
        raise ValueError(f"Number of metadata entries ({len(metadata)}) doesn't match number of vectors ({num_vectors})")

def validate_query_vector(query_vector: np.ndarray, dimension: int) -> np.ndarray:
    """
    Validate and format query vector.
    
    Args:
        query_vector (np.ndarray): Query vector to validate
        dimension (int): Expected dimension
    
    Returns:
        np.ndarray: Formatted query vector
    
    Raises:
        ValueError: If query vector doesn't match expected format
    """
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    
    if query_vector.shape[1] != dimension:
        raise ValueError(f"Expected query vector of dimension {dimension}, got {query_vector.shape[1]}")
    
    return query_vector