import chromadb
from chromadb.config import Settings
import pinecone
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv

load_dotenv()

class VectorStorage:
    def __init__(self, storage_type="chromadb", collection_name="default_collection"):
        self.storage_type = storage_type
        self.collection_name = collection_name
        
        if storage_type == "chromadb":
            self.client = chromadb.Client(Settings(persist_directory="./chroma_db"))
            self.collection = self.client.get_or_create_collection(name=collection_name)
        elif storage_type == "pinecone":
            pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
            self.index = pinecone.Index(collection_name)
        # elif storage_type == "milvus":
        #     # Add Milvus initialization here
        # elif storage_type == "faiss":
        #     # Add FAISS initialization here
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def add_embeddings(self, ids, embeddings, metadatas=None):
        if self.storage_type == "chromadb":
            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
        elif self.storage_type == "pinecone":
            vectors = list(zip(ids, embeddings, metadatas))
            self.index.upsert(vectors=vectors)
        # elif self.storage_type == "milvus":
        #     # Add Milvus insert logic here
        # elif self.storage_type == "faiss":
        #     # Add FAISS insert logic here
    
    def query(self, query_embedding, n_results=5):
        if self.storage_type == "chromadb":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        elif self.storage_type == "pinecone":
            results = self.index.query(query_embedding, top_k=n_results)
            return results
        # elif self.storage_type == "milvus":
        #     # Add Milvus query logic here
        # elif self.storage_type == "faiss":
        #     # Add FAISS query logic here
    
    def delete_embeddings(self, ids):
        if self.storage_type == "chromadb":
            self.collection.delete(ids=ids)
        elif self.storage_type == "pinecone":
            self.index.delete(ids=ids)
        # elif self.storage_type == "milvus":
        #     # Add Milvus delete logic here
        # elif self.storage_type == "faiss":
        #     # Add FAISS delete logic here
    
    def get_embeddings(self, ids):
        if self.storage_type == "chromadb":
            return self.collection.get(ids=ids)
        elif self.storage_type == "pinecone":
            return self.index.fetch(ids=ids)
        # elif self.storage_type == "milvus":
        #     # Add Milvus fetch logic here
        # elif self.storage_type == "faiss":
        #     # Add FAISS fetch logic here
    
    def count_embeddings(self):
        if self.storage_type == "chromadb":
            return self.collection.count()
        elif self.storage_type == "pinecone":
            return self.index.describe_index_stats()['total_vector_count']
        # elif self.storage_type == "milvus":
        #     # Add Milvus count logic here
        # elif self.storage_type == "faiss":
        #     # Add FAISS count logic here

# Example usage
if __name__ == "__main__":
    from utils.huggingface_embed_client import get_embeddings_batch
    
    # Create some sample data
    texts = [
        "This is the first document.",
        "This is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    ids = ["doc1", "doc2", "doc3", "doc4"]
    
    # Generate embeddings
    embeddings = get_embeddings_batch(texts)
    
    # Store embeddings in Chroma DB
    chroma_storage = VectorStorage(storage_type="chromadb", collection_name="sample_collection")
    chroma_storage.add_embeddings(ids=ids, embeddings=embeddings.tolist())
    
    # Query Chroma DB
    query_text = "Which is the first document?"
    query_embedding = get_embeddings_batch([query_text])[0]
    chroma_results = chroma_storage.query(query_embedding)
    print("Chroma DB results:", chroma_results)
    
    # Store embeddings in Pinecone
    pinecone_storage = VectorStorage(storage_type="pinecone", collection_name="sample_collection")
    pinecone_storage.add_embeddings(ids=ids, embeddings=embeddings.tolist())
    
    # Query Pinecone
    pinecone_results = pinecone_storage.query(query_embedding)
    print("Pinecone results:", pinecone_results)
    
    # Count embeddings
    print("Chroma DB count:", chroma_storage.count_embeddings())
    print("Pinecone count:", pinecone_storage.count_embeddings())
    
    # Delete an embedding
    chroma_storage.delete_embeddings(["doc1"])
    pinecone_storage.delete_embeddings(["doc1"])
    
    # Get embeddings
    print("Chroma DB embeddings:", chroma_storage.get_embeddings(["doc2", "doc3"]))
    print("Pinecone embeddings:", pinecone_storage.get_embeddings(["doc2", "doc3"]))