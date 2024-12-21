import unittest
from src.storage.faiss_store import FaissStore
from src.storage.vector_db import VectorDB

class TestFaissStore(unittest.TestCase):
    def setUp(self):
        self.faiss_store = FaissStore()

    def test_add_vectors(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        self.faiss_store.add_vectors(vectors)
        # Add assertions to verify vectors are added correctly

    def test_search(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        self.faiss_store.add_vectors(vectors)
        result = self.faiss_store.search([1.0, 2.0])
        # Add assertions to verify search results

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.vector_db = VectorDB()

    def test_insert(self):
        vector = [1.0, 2.0]
        self.vector_db.insert(vector)
        # Add assertions to verify vector is inserted correctly

    def test_query(self):
        vector = [1.0, 2.0]
        self.vector_db.insert(vector)
        result = self.vector_db.query(vector)
        # Add assertions to verify query results

if __name__ == '__main__':
    unittest.main()