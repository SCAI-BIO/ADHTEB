from abc import ABC
import re

from google import genai
from openai import OpenAI

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

    def get_embedding(self, text: str) -> list:
        """
        Get the embedding of a given text using OpenAI's API.
        """

        try:
            response = self.client.embeddings.create(input=self.sanitize_text(text),
                                                     model=self.model)
            return response.data[0].embedding

        except Exception as e:
            raise RuntimeError(f"Failed to get embedding: {e}")


class GeminiVectorizer(Vectorizer):
    """
    Vectorizer using Gemini's API.
    """

    def __init__(self, model: str = "gemini-embedding", api_key: str = None):
        self.model = model

    def get_embedding(self, text: str) -> list:
        """
        Get the embedding of a given text using Gemini's API.
        """

        if not hasattr(self, 'api_key') or not self.api_key:
            raise ValueError("API key is required to use Gemini's API.")

        try:
            client = genai.Client(api_key="GEMINI_API_KEY")

            result = client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=self.sanitize_text(text))
            return result

        except Exception as e:
            raise RuntimeError(f"Failed to get embedding: {e}")


class HuggingFaceVectorizer(Vectorizer):
    """
    Shared base for HF models.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> list:
        embedding = self.model.encode(self.sanitize_text(text))
        return [float(x) for x in embedding]


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
