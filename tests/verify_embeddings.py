import numpy as np
import os
import json

# Path to the embeddings file
embeddings_file = "C:\\Users\\hasan\\OneDrive\\Documents\\vector-storage-app\\data\\embeddeddata\\article_embeddings_azure_text-embedding-ada-002.npz"

def verify_embeddings(file_path):
    # Load the .npz file
    with np.load(file_path, allow_pickle=True) as data:
        # Get the keys (article IDs)
        article_ids = [key for key in data.keys() if key != '__model_info__']
        
        # Print total number of embeddings
        print(f"Total number of embeddings: {len(article_ids)}")
        
        # Print information for each embedding
        for article_id in article_ids:
            embedding = data[article_id]
            print(f"Article ID: {article_id}")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding mean: {embedding.mean():.4f}")
            print(f"  Embedding std: {embedding.std():.4f}")
            print("---")
        
        # Print model info
        if '__model_info__' in data:
            model_info = json.loads(str(data['__model_info__']))
            print("\nModel Information:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        
        # Print first few and last few article IDs
        print("\nFirst 5 article IDs:")
        print(article_ids[:5])
        print("\nLast 5 article IDs:")
        print(article_ids[-5:])

if __name__ == "__main__":
    if os.path.exists(embeddings_file):
        verify_embeddings(embeddings_file)
    else:
        print(f"File not found: {embeddings_file}")