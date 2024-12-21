import unittest
from src.embeddings.word2vec import Word2Vec
from src.embeddings.transformer import Transformer

class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.word2vec = Word2Vec()
        self.transformer = Transformer()

    def test_word2vec_training(self):
        # Sample training data
        sentences = [["hello", "world"], ["machine", "learning"]]
        self.word2vec.train(sentences)
        vector = self.word2vec.get_vector("hello")
        self.assertIsNotNone(vector)

    def test_transformer_encoding(self):
        # Sample input sequence
        input_sequence = ["this", "is", "a", "test"]
        embeddings = self.transformer.encode(input_sequence)
        self.assertEqual(len(embeddings), len(input_sequence))

    def test_transformer_get_embedding(self):
        input_sequence = ["this", "is", "another", "test"]
        self.transformer.encode(input_sequence)
        embedding = self.transformer.get_embedding("this")
        self.assertIsNotNone(embedding)

if __name__ == '__main__':
    unittest.main()