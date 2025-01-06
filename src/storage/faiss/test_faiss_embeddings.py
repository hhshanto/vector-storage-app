# src/storage/faiss/test_faiss_embeddings.py
import numpy as np
import sys
import os
from store import FAISSStore

def load_embeddings(npz_file_path):
    """
    Load embeddings from npz file
    """
    try:
        print(f"\nLoading embeddings from: {npz_file_path}")
        data = np.load(npz_file_path, allow_pickle=True)
        
        # Get the article IDs (excluding model info)
        article_ids = [key for key in data.keys() if key != '__model_info__']
        print(f"Total number of articles: {len(article_ids)}")
        
        # Get the first embedding to determine dimension
        first_embedding = data[article_ids[0]]
        dimension = first_embedding.shape[0]
        print(f"Embedding dimension: {dimension}")
        
        # Stack all embeddings into a single array
        embeddings = np.vstack([data[article_id] for article_id in article_ids])
        print(f"Combined embeddings shape: {embeddings.shape}")
        
        return embeddings, article_ids
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None

def test_embeddings_store():
    """
    Test FAISS store with Azure OpenAI embeddings
    """
    # Path to embeddings file
    npz_file = r"C:\Users\hasan\OneDrive\Documents\vector-storage-app\data\EmbeddedData\article_embeddings_azure_text-embedding-ada-002.npz"
    
    print("\nStarting FAISS store test with Azure OpenAI embeddings...")
    
    # Load embeddings
    vectors, article_ids = load_embeddings(npz_file)
    if vectors is None:
        return
    
    try:
        # Initialize FAISS store with the correct dimension
        dimension = vectors.shape[1]
        store = FAISSStore(dimension=dimension)
        print("✓ Successfully created FAISS store")
        
        # Create metadata with original article IDs
        metadata = [
            {
                "article_id": article_id,
                "original_index": i
            } 
            for i, article_id in enumerate(article_ids)
        ]
        
        # Add vectors
        store.add_vectors(vectors, article_ids, metadata)
        print(f"✓ Successfully added {len(article_ids)} vectors")
        
        # Test search with first vector
        print("\nTesting similarity search...")
        query = vectors[0:1]
        results = store.search(query, k=3)
        
        print("\nSearch results for first article:")
        for id_str, distance, metadata in results:
            print(f"\nArticle ID: {id_str}")
            print(f"Distance: {distance:.4f}")
            print(f"Original Index: {metadata['original_index']}")
        
        # Try a random vector query
        random_idx = np.random.randint(0, len(article_ids))
        query = vectors[random_idx:random_idx+1]
        results = store.search(query, k=3)
        
        print(f"\nSearch results for random article (index {random_idx}):")
        for id_str, distance, metadata in results:
            print(f"\nArticle ID: {id_str}")
            print(f"Distance: {distance:.4f}")
            print(f"Original Index: {metadata['original_index']}")
        
        print("\nAll tests passed!")
        
        # Save the index
        save_path = "azure_embeddings_index"
        store.save(save_path)
        print(f"\n✓ Saved index to {save_path}")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_embeddings_store()