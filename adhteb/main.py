import os
from adhteb.benchmarking import Benchmark
from adhteb.vectorizers import *

import logging

logging.basicConfig(
    level=logging.INFO,  # Still allows INFO level from your logger
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

for noisy_logger in ["httpx", "openai", "urllib3"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logging.getLogger("benchmarking").setLevel(logging.INFO)

# load the OpenAI API key from env
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GENAI_API_KEY")

openai_vectorizer = OpenAIVectorizer(api_key=openai_api_key)
gemini_vectorizer = GeminiVectorizer(api_key=gemini_api_key)

#linq_vectorizer = LinqEmbedMistralVectorizer()
#mini_lm_vectorizer = AllMiniLMVectorizer()
#qwen_vectorizer = Qwen38BVectorizer()

vectorizers = [gemini_vectorizer]

for vectorizer in vectorizers:
    print(f"Running benchmark for {vectorizer.model_name}...")
    benchmark = Benchmark(vectorizer=vectorizer)
    benchmark.run()
    print(benchmark.results_summary())
    benchmark.results_geras.save_pr_curve()
    benchmark.results_emif.save_pr_curve()
    benchmark.results_prevent_dementia.save_pr_curve()
    benchmark.results_prevent_ad.save_pr_curve()
    benchmark.save(f'results/{vectorizer.model_name}.pkl')
