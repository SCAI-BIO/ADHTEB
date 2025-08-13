from abc import ABC
import re
import time
from typing import List

from openai import OpenAI
from google import genai

from sentence_transformers import SentenceTransformer


class Vectorizer(ABC):
    """
    Abstract base class for vectorizers.
    """

    @property
    def model_name(self) -> str:
        class_name = self.__class__.__name__
        model_name = class_name.replace("Vectorizer", "")
        return model_name

    def get_embedding(self, text: str) -> list:
        """
        Get the embedding of a given text.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_embeddings_batch(self, texts: List[str]) -> List[list]:
        """
        Get embeddings for a list of texts in a batch.
        Default implementation calls get_embedding for each text.
        Subclasses can override for optimized batch processing.
        """
        return [self.get_embedding(text) for text in texts]

    def sanitize_text(self, text: str) -> str:
        """
        Clean and normalize input text for embedding models:
        - Unicode normalization (NFKC)
        - Remove HTML tags and URLs
        - Normalize whitespace and line breaks
        - Remove non-printable/control characters
        - Optionally strip or lower-case text
        - Collapse multiple spaces
        """
        try:
            # Remove control characters and normalize line breaks
            text = re.sub(r'[\r\n\t]+', ' ', text)

            # Remove everything except common punctuation
            text = re.sub(r'[^\w\s\.,!\?;:\'\"\-]', ' ', text)

            # Lowercase
            text = text.lower()

            # Collapse extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

        except Exception as e:
            raise ValueError(f"Error sanitizing text {text}: {e}")

        return text


class OpenAIVectorizer(Vectorizer):
    """
    Vectorizer using OpenAI's API.
    """

    def __init__(self, model: str = "text-embedding-3-large", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.rate_limit_per_minute = 3000  # OpenAI's default rate limit for embedding models
        self.last_request_time = 0

    def get_embedding(self, text: str) -> list:
        """
        Get the embedding of a given text using OpenAI's API.
        """
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[list]:
        """
        Get embeddings for a list of texts using OpenAI's API in a batch.
        """
        sanitized_texts = [self.sanitize_text(text) for text in texts]

        # Implement a basic rate limiting mechanism
        current_time = time.time()
        time_elapsed = current_time - self.last_request_time
        min_time_between_requests = 60 / self.rate_limit_per_minute
        if time_elapsed < min_time_between_requests:
            time.sleep(min_time_between_requests - time_elapsed)
        self.last_request_time = time.time()

        try:
            response = self.client.embeddings.create(input=sanitized_texts,
                                                     model=self.model,
                                                     encoding_format='float')
            return [[float(x) for x in data.embedding] for data in response.data]

        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings for batch: {e}")


class GeminiVectorizer(Vectorizer):
    """
    Vectorizer using Gemini's API.
    """

    def __init__(self, model: str = "embedding-001", api_key: str = None):
        self.model = model
        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError("API key is required to initialize GeminiVectorizer.")
        self.rate_limit_per_minute = 150
        self.last_request_time = 0

    def get_embedding(self, text: str) -> list:
        """
        Get the embedding of a given text using Gemini's API.
        """
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[list]:
        """
        Get embeddings for a list of texts using Gemini's API in a batch.
        """
        if not hasattr(self, 'api_key') or not self.api_key:
            raise ValueError("API key is required to use Gemini's API.")

        sanitized_texts = [self.sanitize_text(text) for text in texts]

        # Implement a basic rate limiting mechanism
        current_time = time.time()
        time_elapsed = current_time - self.last_request_time
        min_time_between_requests = 60 / self.rate_limit_per_minute
        if time_elapsed < min_time_between_requests:
            time.sleep(min_time_between_requests - time_elapsed)
        self.last_request_time = time.time()

        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=sanitized_texts,
            )
            embeddings = [[float(x) for x in embedding.values] for embedding in response.embeddings]

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings for batch: {e}")


class HuggingFaceVectorizer(Vectorizer):
    """
    Shared base for HF models.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> list:
        embedding = self.model.encode(self.sanitize_text(text))
        return [float(x) for x in embedding]

    def get_embeddings_batch(self, texts: List[str]) -> List[list]:
        sanitized_texts = [self.sanitize_text(text) for text in texts]
        embeddings = self.model.encode(sanitized_texts)
        return [[float(x) for x in emb] for emb in embeddings]


class LinqEmbedMistralVectorizer(HuggingFaceVectorizer):
    """
    Linq-Embed-Mistral from Baichuan
    """

    def __init__(self):
        super().__init__("Linq-AI-Research/Linq-Embed-Mistral")


class Qwen38BVectorizer(HuggingFaceVectorizer):
    """
    Qwen3, 2nd best performning model after gemini from MTEB
    """

    def __init__(self):
        super().__init__("Qwen/Qwen3-Embedding-8B")


class AllMiniLMVectorizer(HuggingFaceVectorizer):
    """
    all-MiniLM-L6-v2 - most used vectorizer from HF
    """

    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2")