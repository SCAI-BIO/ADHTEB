import os
import unittest
import random
import warnings

from typing import List
from adhteb.benchmarking import Benchmark
from adhteb.vectorizers import Vectorizer

import importlib.resources as pkg_resources


class MockVectorizer(Vectorizer):

    def __init__(self, model_name: str = "mock-embedding", dim: int = 32):
        self._model_name = model_name
        self._dim = dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_embedding(self, text: str) -> list:
        """Return a random embedding of fixed dimension."""
        return [random.random() for _ in range(self._dim)]

    def get_embeddings_batch(self, texts: List[str]) -> List[list]:
        """Return a list of random embeddings for a batch of texts."""
        return [[random.random() for _ in range(self._dim)] for text in texts]



class BenchmarkTest(unittest.TestCase):

    def setUp(self):
        self.vectorizer = MockVectorizer()
        data_path_public = pkg_resources.files('adhteb.data.cohorts.public')
        cohorts_files = [entry for entry in data_path_public.iterdir()
                         if entry.is_file() and entry.name.endswith(".csv")]
        self.n_cohorts_public = len(cohorts_files)
        data_path_private = pkg_resources.files('adhteb.data.cohorts.private')
        private_cohorts_files = [entry for entry in data_path_private.iterdir()
                                 if entry.is_file() and entry.name.endswith(".csv")]
        self.n_cohorts_private = len(private_cohorts_files) + 1
        # +1 for combined GERAS studies in geras subdir

    def test_run_public_only(self):
        try:
            benchmark = Benchmark(vectorizer=self.vectorizer)
            benchmark.run()
            benchmark.results_summary()
            self.assertEqual(self.n_cohorts_public, len(benchmark.results))
        except Exception as e:
            self.fail(f"run() raised an exception: {e}")

    def test_run_private_and_public(self):
        try:
            decryption_key = os.getenv("ADHTEB_DECRYPT_KEY")
            # for automated dependency updates, secrets are hidden -> skip this test to enable these updates
            if not decryption_key:
                warnings.warn("Environment variable 'ADHTEB_DECRYPT_KEY' not set — skipping test.")
                return
            benchmark = Benchmark(vectorizer=self.vectorizer, include_private=True, decryption_key=decryption_key)
            benchmark.run()
            benchmark.results_summary()
            self.assertEqual(self.n_cohorts_public + self.n_cohorts_private, len(benchmark.results))
        except Exception as e:
            self.fail(f"run() raised an exception: {e}")
