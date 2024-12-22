import sys
import os
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.utils.huggingface_embed_client import get_embedding, get_embeddings_batch, DEFAULT_MODEL

def load_preprocessed_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_article_embedding(article_data, model_name=DEFAULT_MODEL):
    """
    Create embedding for the preprocessed article along with its ID.
    
    Args:
    article_data (dict): The preprocessed article data containing 'id' and 'processed_article'.
    model_name (str): The name of the model to use for embedding.
    
    Returns:
    dict: A dictionary containing the article ID, its embedding, and model info.
    """
    article_id = article_data['id']
    processed_article = article_data['processed_article']
    
    embedding, model_info = get_embedding(processed_article, model_name)
    
    return {
        'id': article_id,
        'embedding': embedding,
        'model_info': model_info
    }

def create_embeddings_for_dataset(preprocessed_data, batch_size=32, model_name=DEFAULT_MODEL):
    """
    Create embeddings for all articles in the preprocessed dataset using batch processing.
    
    Args:
    preprocessed_data (list): List of preprocessed article dictionaries.
    batch_size (int): Number of articles to process in each batch.
    model_name (str): The name of the model to use for embedding.
    
    Returns:
    tuple: (list of dicts, dict) Article embeddings and model info.
    """
    article_embeddings = []
    model_info = None
    for i in range(0, len(preprocessed_data), batch_size):
        batch = preprocessed_data[i:i+batch_size]
        ids = [article['id'] for article in batch]
        texts = [article['processed_article'] for article in batch]
        
        embeddings, batch_model_info = get_embeddings_batch(texts, model_name)
        
        if model_info is None:
            model_info = batch_model_info
        
        for id, embedding in zip(ids, embeddings):
            article_embeddings.append({
                'id': id,
                'embedding': embedding
            })
    
    return article_embeddings, model_info

def save_embeddings(embeddings, model_info, output_file):
    """
    Save embeddings to a numpy file.
    
    Args:
    embeddings (list): List of dictionaries containing article IDs and their embeddings.
    model_info (dict): Information about the model used for embedding.
    output_file (str): Path to the output file.
    """
    embedding_dict = {item['id']: item['embedding'] for item in embeddings}
    np.savez(output_file, **embedding_dict, __model_info__=json.dumps(model_info))

if __name__ == "__main__":
    preprocessed_data_file = os.path.join(project_root, "data", "preprocessed_passages.json")
    
    preprocessed_data = load_preprocessed_data(preprocessed_data_file)
    
    article_embeddings, model_info = create_embeddings_for_dataset(preprocessed_data)
    
    for article_embedding in article_embeddings:
        print(f"Article ID: {article_embedding['id']}")
        print(f"Embedding shape: {article_embedding['embedding'].shape}")
        print("---")
    
    print(f"Model Name: {model_info['model_name']}")
    print(f"Model Type: {model_info['model_type']}")
    print(f"Embedding Dimension: {model_info['embedding_dim']}")
    
    # Create a filename that includes the model name
    model_name_safe = model_info['model_name'].replace('/', '_')  # Replace '/' with '_' for safe filenames
    output_filename = f"article_embeddings_{model_name_safe}.npz"
    output_file = os.path.join(project_root, "data", "embeddeddata", output_filename)
    
    save_embeddings(article_embeddings, model_info, output_file)
    print(f"Embeddings saved to {output_file}")
