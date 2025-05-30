
import os
from typing import Dict, List
from typing import Dict, List
from benchmarking import Benchmark
from vectorizers import OpenAIVectorizer, LinqEmbedMistralVectorizer
import logging

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set the default logging level to WARNING for all loggers
logging.getLogger().setLevel(logging.WARNING)

# Set the logging level to INFO for the benchmarking class
logging.getLogger('benchmarking').setLevel(logging.INFO)

# load the OpenAI API key from env
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_vectorizer = OpenAIVectorizer(api_key=openai_api_key)
#linq_embed_mistral = LinqEmbedMistralVectorizer()

vectorizers = [openai_vectorizer]

for vectorizer in vectorizers:
    print(f"Running benchmark for {vectorizer.model_name}...")
    benchmark : Dict[str, List[float]]= Benchmark(vectorizer=vectorizer)
    accuracies: Dict[str, List[float]] = benchmark.get_accuracies()
    print(f"Accuracies for {vectorizer.model_name}: {accuracies}")
    
