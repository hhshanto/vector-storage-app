class Word2Vec:
    def __init__(self, vector_size=100, window=5, min_count=1, sg=0):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None

    def train(self, sentences):
        # Implement training logic here
        pass

    def get_vector(self, word):
        # Implement logic to retrieve the embedding of a specific word
        pass