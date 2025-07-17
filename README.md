# Alzheimer's Disease Harmonization Text Embedding Benchmark
# Summary

This repository provides a benchmark for evaluating text embeddings in the context of Alzheimer's disease research. 

# Installation
```bash
pip install adhteb
```

# Usage

## Import a model

Models that are published on huggingface can be directly imported using the HuggingFaceVectorizer class.

```python
from adhteb import HuggingFaceVectorizer

vectorizer = HuggingFaceVectorizer(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
```

Alternatively, you can implement your own vectorizer by implementing the `get_embedding` method of the base class.

```python
from adhteb import Vectorizer

class MyVectorizer(Vectorizer):
    def get_embedding(self, text: str) -> list[float]:
        # Implement your embedding logic here
        my_vector = []
        return my_vector
```
## Running the benchmark

You can run the benchmark and display the results using only a few lines of code.

```python
from adhteb import Benchmark

    benchmark = Benchmark(vectorizer=vectorizer)
    benchmark.run()
    print(benchmark.results_summary())
```
```commandline
+------------------+-------+--------------------+
|                  | AUPRC | Zero-shot Accuracy |
+------------------+-------+--------------------+
|      GERAS       | 0.35  |        0.65        |
| PREVENT Dementia | 0.19  |        0.48        |
|    PREVENT AD    | 0.22  |        0.39        |
|       EMIF       | 0.29  |        0.54        |
+------------------+-------+--------------------+
Aggregate Score: 0.39
```

## Publishing your results

You can check how your results compare to other models on the public leaderboard here:
[https:adhteb.scai.fraunhofer.de](https:adhteb.scai.fraunhofer.de)

You are also able to publish your benchmark results together with metadata on yout tested model:

```python
from adhteb import Benchmark, ModelMetadata

model_name= "my-model-name"
url="https://huggingface.co/my-model-name"

model_metadata = ModelMetadata(model_name=model_name, url=url)
benchmark.publish(model_metadata=model_metadata)
```

