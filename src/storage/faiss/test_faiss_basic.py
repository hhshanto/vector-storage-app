# src/storage/faiss/test_faiss_basic.py
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.storage.faiss.store import FAISSStore

def test_basic_faiss_store():
    """
    Test basic FAISS store functionality
    """
    # Test parameters
    dimension = 3
    num_vectors = 4
    
    print("\nStarting FAISS store basic test...")
    
    # Create sample vectors
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float32)
    
    # Create sample IDs
    ids = [f"vec_{i}" for i in range(num_vectors)]
    
    # Create sample metadata
    metadata = [
        {"description": f"Vector {i}"} for i in range(num_vectors)
    ]
    
    try:
        # Initialize FAISS store
        store = FAISSStore(dimension=dimension)
        print("✓ Successfully created FAISS store")
        
        # Add vectors
        store.add_vectors(vectors, ids, metadata)
        print(f"✓ Successfully added {num_vectors} vectors")
        
        # Verify the number of vectors in the index
        assert store.index.ntotal == num_vectors
        print(f"✓ Verified number of vectors: {store.index.ntotal}")
        
        # Verify metadata storage
        for id_str in ids:
            assert id_str in store.metadata_dict
        print("✓ Verified metadata storage")
        
        # Print some metadata to verify content
        print("\nSample metadata:")
        print(store.metadata_dict["vec_0"])
        
        print("\nAll basic tests passed!")
        
        # Call test_search function
        test_search(store, vectors)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise

def test_search(store, vectors):
    print("\nTesting search functionality...")
    
    # Use the first vector as a query
    query = vectors[0:1]  # Keep 2D shape
    results = store.search(query, k=2)
    
    print(f"Search results for vector [1.0, 0.0, 0.0]:")
if __name__ == "__main__":
    test_basic_faiss_store()
