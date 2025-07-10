import os
from benchmarking import Benchmark
from vectorizers import *
import logging

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Set the default logging level to WARNING for all loggers
logging.getLogger().setLevel(logging.WARNING)

# Set the logging level to INFO for the benchmarking class
logging.getLogger("benchmarking").setLevel(logging.INFO)

# load the OpenAI API key from env
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_vectorizer = OpenAIVectorizer(api_key=openai_api_key)

vectorizers = [openai_vectorizer]

for vectorizer in vectorizers:
    print(f"Running benchmark for {vectorizer.model_name}...")
    benchmark = Benchmark(vectorizer=vectorizer)
    benchmark.run()
    print(benchmark.results_summary())
    benchmark.results_geras.save_pr_curve()
    benchmark.results_a4.save_pr_curve()
    benchmark.results_prevent_dementia.save_pr_curve()
    benchmark.results_aibl.save_pr_curve()
    benchmark.publish()