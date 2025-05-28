
import os
from benchmarking import Benchmark
from vectorizers import OpenAIVectorizer, LinqEmbedMistralVectorizer
import logging

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# load the OpenAI API key from env
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_vectorizer = OpenAIVectorizer(api_key=openai_api_key)
#linq_embed_mistral = LinqEmbedMistralVectorizer()

vectorizers = [openai_vectorizer]

for vectorizer in vectorizers:
    print(f"Running benchmark for {vectorizer.model_name}...")
    benchmark = Benchmark(vectorizer=vectorizer)
    accuracies = benchmark.get_accuracies()
    print(f"Accuracies for {vectorizer.model_name}: {accuracies}")