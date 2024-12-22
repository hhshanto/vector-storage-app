import os
import sys
from typing import List, Tuple, Dict
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.utils.azure_openai_api_client import get_gpt_client

AZURE_EMBEDDING_MODELS = {
    "text-embedding-ada-002": "General purpose embedding model, 1536 dimensions"
}

# Default model
DEFAULT_MODEL = "text-embedding-ada-002"

def get_embedding(text: str, model_name: str = DEFAULT_MODEL) -> Tuple[np.ndarray, Dict]:
    """
    Generate embedding for the given text using the specified Azure OpenAI model.
    
    Args:
    text (str): The input text to embed.
    model_name (str): The name of the model to use for embedding.
    
    Returns:
    tuple: (numpy.ndarray, dict) The embedding vector and model info.
    """
    client = get_gpt_client()
    
    response = client.embeddings.create(
        input=text,
        model=model_name
    )
    
    embedding = np.array(response.data[0].embedding)
    model_info = {
        'model_name': model_name,
        'embedding_dim': embedding.shape[0],
        'model_type': 'azure_openai'
    }
    return embedding, model_info

def get_embeddings_batch(texts: List[str], model_name: str = DEFAULT_MODEL) -> Tuple[List[np.ndarray], Dict]:
    """
    Generate embeddings for a batch of texts using the specified Azure OpenAI model.
    
    Args:
    texts (list of str): The input texts to embed.
    model_name (str): The name of the model to use for embedding.
    
    Returns:
    tuple: (list of numpy.ndarray, dict) The embedding vectors and model info.
    """
    client = get_gpt_client()
    
    response = client.embeddings.create(
        input=texts,
        model=model_name
    )
    
    embeddings = [np.array(item.embedding) for item in response.data]
    model_info = {
        'model_name': model_name,
        'embedding_dim': embeddings[0].shape[0],
        'model_type': 'azure_openai'
    }
    return embeddings, model_info

def list_available_models():
    """
    List all available Azure OpenAI embedding models with their descriptions.
    """
    for model, description in AZURE_EMBEDDING_MODELS.items():
        print(f"{model}: {description}")

def change_default_model(model_name: str):
    """
    Change the default embedding model.
    
    Args:
    model_name (str): The name of the new default model.
    """
    global DEFAULT_MODEL
    if model_name in AZURE_EMBEDDING_MODELS:
        DEFAULT_MODEL = model_name
        print(f"Default model changed to: {DEFAULT_MODEL}")
    else:
        print(f"Model '{model_name}' not found in available models.")
        list_available_models()

# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample sentence for embedding."
    embedding, model_info = get_embedding(sample_text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Model info: {model_info}")
    
    print("\nAvailable models:")
    list_available_models()