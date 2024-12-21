# README.md

# Vector Storage and Embedding Techniques

This project explores various vector storage and embedding techniques, providing implementations for popular algorithms and storage solutions.

## Overview

The Vector Storage App is designed to help users understand and implement different embedding techniques, such as Word2Vec and transformer-based embeddings, as well as how to efficiently store and retrieve these vectors using various storage solutions.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hhshanto/vector-storage-app.git
cd vector-storage-app
pip install -r requirements.txt
```

## Usage

### Embeddings

- **Word2Vec**: Train a Word2Vec model and retrieve word embeddings.
- **Transformer**: Use transformer-based models to encode sequences and obtain embeddings.

### Storage

- **FaissStore**: Utilize Facebook's Faiss library for efficient similarity search.
- **VectorDB**: Interface for storing and querying vectors.

## Running Tests

To ensure everything is working correctly, run the unit tests:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.