# Vector Storage and Embedding Techniques Lab

A comprehensive exploration of vector storage solutions and embedding techniques commonly used in modern AI applications, with a focus on Azure OpenAI and Hugging Face models.

## Project Goal

This project aims to provide a hands-on learning experience and demonstration of various vector storage and embedding techniques. The goal is to understand the strengths and weaknesses of different approaches, their performance characteristics, and practical applications in AI and machine learning workflows.

## Project Overview

This project demonstrates practical implementations of:

### 1. Embedding Techniques
- Azure OpenAI embeddings
- Hugging Face Transformer-based embeddings
- Custom embeddings

### 2. Vector Storage Solutions
- FAISS (Facebook AI Similarity Search)
- Potential for integration with other solutions like Chroma DB, Pinecone, or Milvus

### 3. Similarity Search Methods
- Cosine Similarity
- Euclidean Distance
- Dot Product

## Technical Skills Demonstrated

- Vector Operations
- Efficient Vector Storage
- Similarity Search Algorithms
- Integration with Cloud Services (Azure OpenAI)
- Performance Optimization

## Installation

```bash
git clone https://github.com/your-username/vector-storage-app.git
cd vector-storage-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
vector-storage-app/
├── src/
│   ├── embeddings/
│   │   ├── create_embedding_azure.py
│   │   └── create_embedding_huggingface.py
│   ├── storage/
│   │   └── (future vector storage implementations)
│   └── utils/
│       ├── azure_openai_api_client.py
│       └── azure_openai_embed_client.py
├── tests/
│   └── verify_embeddings.py
├── data/
│   ├── preprocessed_passages.json
│   └── embeddeddata/
│       └── (generated embedding files)
├── notebooks/
│   └── (Jupyter notebooks for examples and analysis)
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage Examples

### 1. Creating Embeddings with Azure OpenAI

```python
from src.embeddings.create_embedding_azure import create_embeddings_for_dataset

preprocessed_data = load_preprocessed_data("path/to/preprocessed_data.json")
embeddings, model_info = create_embeddings_for_dataset(preprocessed_data)
```

### 2. Creating Embeddings with Hugging Face

```python
from src.embeddings.create_embedding_huggingface import create_embeddings_for_dataset

preprocessed_data = load_preprocessed_data("path/to/preprocessed_data.json")
embeddings, model_info = create_embeddings_for_dataset(preprocessed_data)
```

### 3. Verifying Embeddings

```python
from tests.verify_embeddings import verify_embeddings

verify_embeddings("path/to/embeddings.npz")
```

## Embedding Models Used

1. Azure OpenAI: text-embedding-ada-002
2. Hugging Face: (<br />
    'sentence-transformers/all-MiniLM-L6-v2': 'Compact and efficient model for sentence embeddings',<br />
    'sentence-transformers/all-mpnet-base-v2': 'High-performance model for sentence embeddings',<br />
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'Multilingual model supporting 50+ languages',<br />
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens': 'DistilBERT-based model for semantic similarity',<br />
    'openai-gpt': 'OpenAI\'s GPT model for general-purpose text embeddings',<br />
    'bert-base-uncased': 'Classic BERT model, widely used baseline',<br />
    'roberta-base': 'Improved version of BERT',<br />
    'xlm-roberta-base': 'Multilingual version of RoBERTa',<br />
    'allenai/scibert_scivocab_uncased': 'Specialized BERT model for scientific text',<br />
    'microsoft/deberta-base': 'Enhanced BERT model with disentangled attention'<br />
)

## Results and Findings



## Challenges and Learnings

During this project, some challenges and learnings included:

1. Integrating Azure OpenAI API for embedding generation
2. Handling batch processing of embeddings for efficiency
3. Comparing embeddings from different sources (Azure vs. Hugging Face)


## Resources

- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any inquiries or collaboration opportunities, please reach out:

- GitHub: [hhshanto](https://github.com/hhshanto)
- LinkedIn: [mhasan-shanto](https://www.linkedin.com/in/mhasan-shanto/)

## License

MIT License
