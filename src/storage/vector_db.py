class VectorDB:
    def __init__(self):
        self.vectors = {}

    def insert(self, key, vector):
        """Insert a vector into the database."""
        self.vectors[key] = vector

    def query(self, key):
        """Retrieve a vector from the database."""
        return self.vectors.get(key, None)