from transformers import AutoTokenizer, AutoModel
import torch
import json

# Dictionary of model names and their descriptions
EMBEDDING_MODELS = {
    'sentence-transformers/all-MiniLM-L6-v2': 'Compact and efficient model for sentence embeddings',
    'sentence-transformers/all-mpnet-base-v2': 'High-performance model for sentence embeddings',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'Multilingual model supporting 50+ languages',
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens': 'DistilBERT-based model for semantic similarity',
    'openai-gpt': 'OpenAI\'s GPT model for general-purpose text embeddings',
    'bert-base-uncased': 'Classic BERT model, widely used baseline',
    'roberta-base': 'Improved version of BERT',
    'xlm-roberta-base': 'Multilingual version of RoBERTa',
    'allenai/scibert_scivocab_uncased': 'Specialized BERT model for scientific text',
    'microsoft/deberta-base': 'Enhanced BERT model with disentangled attention'
}

# Default model
DEFAULT_MODEL = 'sentence-transformers/all-mpnet-base-v2'

def get_embedding(text, model_name=DEFAULT_MODEL):
    """
    Generate embedding for the given text using the specified model.
    
    Args:
    text (str): The input text to embed.
    model_name (str): The name of the model to use for embedding.
    
    Returns:
    tuple: (numpy.ndarray, dict) The embedding vector and model info.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    model_info = {
        'model_name': model_name,
        'embedding_dim': embedding.shape[0],
        'model_type': model.config.model_type
    }
    return embedding, model_info

def list_available_models():
    """
    List all available embedding models with their descriptions.
    """
    for model, description in EMBEDDING_MODELS.items():
        print(f"{model}: {description}")

def change_default_model(model_name):
    """
    Change the default embedding model.
    
    Args:
    model_name (str): The name of the new default model.
    """
    global DEFAULT_MODEL
    if model_name in EMBEDDING_MODELS:
        DEFAULT_MODEL = model_name
        print(f"Default model changed to: {DEFAULT_MODEL}")
    else:
        print(f"Model '{model_name}' not found in available models.")
        list_available_models()


def get_embeddings_batch(texts, model_name=DEFAULT_MODEL):
    """
    Generate embeddings for a batch of texts using the specified model.
    
    Args:
    texts (list of str): The input texts to embed.
    model_name (str): The name of the model to use for embedding.
    
    Returns:
    tuple: (list of numpy.ndarray, dict) The embedding vectors and model info.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    model_info = {
        'model_name': model_name,
        'embedding_dim': embeddings.shape[1],
        'model_type': model.config.model_type
    }
    return embeddings, model_info


# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample sentence for embedding."
    embedding = get_embedding(sample_text)
    print(f"Embedding shape: {embedding.shape}")
    
    print("\nAvailable models:")
    list_available_models()