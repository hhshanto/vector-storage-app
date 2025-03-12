# Vector Storage and Embedding Techniques Lab

A comprehensive platform for exploring vector embeddings and efficient storage techniques for text data retrieval.

## Overview

This project implements and compares various vector embedding strategies and storage solutions for semantic search applications. It provides a framework to preprocess text data, generate vector embeddings, store them efficiently, and implement similarity search methods.

## Features

- **Multiple Embedding Techniques**:
  - Azure OpenAI API embeddings
  - Hugging Face transformers models
  - Support for dimensionality experiments

- **Vector Storage Solutions**:
  - FAISS (Facebook AI Similarity Search) implementation
  - Optimized index structures for efficient retrieval
  - Metadata management and persistence

- **Data Processing**:
  - Text preprocessing and normalization
  - Passage extraction and formatting
  - Question-answer pair handling

- **Search Capabilities**:
  - Semantic similarity search
  - Nearest neighbor algorithms
  - Performance-optimized retrieval

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
├── src/                          # Core source code
│   ├── embeddings/               # Embedding generation modules
│   │   ├── create_embedding_azure.py
│   │   └── create_embedding_huggingface.py
│   ├── storage/                  # Vector storage implementations
│   │   └── (FAISS and other storage solutions)
│   └── utils/                    # Utility functions
│       ├── azure_openai_api_client.py
│       └── azure_openai_embed_client.py
├── tests/                        # Unit and integration tests
│   ├── test_embeddings.py
│   └── test_storage.py
├── data/                         # Dataset files
│   ├── preprocessed_passages.json
│   ├── race_qa_pairs.txt
│   ├── selected_passages/        # Individual passage files
│   └── EmbeddedData/             # Generated embedding files
├── notebooks/                    # Jupyter notebooks
│   ├── embedding_dimensions.ipynb
│   └── huggingface_embedding_models.ipynb
├── docs/                         # Documentation
│   ├── dataset_summary.txt
│   └── FAISS.md                  # FAISS implementation details
├── .env                          # Environment variables (API keys)
├── requirements.txt              # Project dependencies
└── CHANGELOG.md                  # Development progress tracking
```

## Usage Examples

### Generating Embeddings

```python
from src.embeddings.create_embedding_azure import generate_embeddings

# Generate embeddings for a text passage
text = "This is a sample passage for embedding generation"
embedding = generate_embeddings(text)
```

### Vector Storage and Retrieval

```python
from src.storage.faiss_storage import FAISSIndex

# Create and save an index
index = FAISSIndex(dimension=1536)
index.add_vectors(embeddings, metadata)
index.save("data/EmbeddedData/my_index.faiss")

# Search for similar vectors
results = index.search(query_vector, k=5)
```

## Datasets

The project works with text passages from the RACE dataset (Reading Comprehension from Examinations), which includes:
- Reading passages
- Multiple-choice questions
- Answer options
- Correct answers

## Documentation

For detailed information about FAISS implementation, refer to FAISS.md.

## Future Work

Based on the project changelog, the following features are planned:
- Additional embedding techniques (Word2Vec, Doc2Vec, FastText)
- More vector storage solutions (ChromaDB, Pinecone, Milvus)
- Advanced similarity search methods
- Performance benchmarking and optimization
- Web interface for demonstration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License