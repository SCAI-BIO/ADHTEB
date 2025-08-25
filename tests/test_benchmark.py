import os
import unittest
from typing import List

from adhteb.benchmarking import Benchmark
from adhteb.vectorizers import Vectorizer


class MockVectorizer(Vectorizer):

    def __init__(self, model_name="MockModel"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_embedding(self, text: str) -> list:
        return [float(len(text))] * 768

    def get_embeddings_batch(self, texts: List[str]) -> List[list]:
        return [[float(len(text))] * 768 for text in texts]


class BenchmarkTest(unittest.TestCase):

    def setUp(self):
        self.vectorizer = MockVectorizer()

    def test_run(self):
        try:
            decryption_key = os.getenv("ADHTEB_DECRYPT_KEY")
            benchmark = Benchmark(vectorizer=self.vectorizer, include_private=True, decryption_key=decryption_key)
            benchmark.run()
        except Exception as e:
            self.fail(f"run() raised an exception: {e}")
