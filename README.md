# Alzheimer's Disease Harmonization Text Embedding Benchmark

<a href="https://doi.org/10.5281/zenodo.16027340"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16027340-blue.svg" alt="DOI"></a>&nbsp;<a href="https://github.com/SCAI-BIO/ADHTEB/actions/workflows/tests.yaml"><img src="https://github.com/SCAI-BIO/ADHTEB/actions/workflows/tests.yaml/badge.svg" alt="tests"></a>&nbsp;<a href="https://pypi.org/project/adhteb/"><img src="https://img.shields.io/pypi/v/adhteb" alt="version"></a>&nbsp;<a href="https://pepy.tech/projects/adhteb"><img src="https://static.pepy.tech/personalized-badge/adhteb?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads" alt="PyPI Downloads"></a>

# About

ADHTEB (Alzheimer’s Disease Harmonization Text Embedding Benchmark) is a Python package designed to evaluate the performance of text-embedding models in harmonizing variable descriptions from diverse cohorts in the context of Alzheimer’s disease. 

As general purpose benchmarks often lack domain-specific evaluation for clinical data, this benchmark is specifically designed to evaluate the performance of embedding models for harmonization or clustering of data descriptions in a clinical setting.

For implementation details and the initial assessment, see the  [the associated paper](https://www.sciencedirect.com/science/article/pii/S2274580725003619).

# Installation
```bash
pip install adhteb
```

# Usage

## Import a model

Models that are published on huggingface can be directly imported using the HuggingFaceVectorizer class.

```python
from adhteb import HuggingFaceVectorizer
from sentence_transformers import SentenceTransformer

# pass model name or SentenceTransformer object instance (in case of additional params)
vectorizer = HuggingFaceVectorizer(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# alternative
sentence_transformer = SentenceTransformer(...)
vectorizer = HuggingFaceVectorizer(sentence_transformer)
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
[https://adhteb.scai.fraunhofer.de](https://adhteb.scai.fraunhofer.de)

You are also able to publish your benchmark results together with metadata on your tested model:

```python
from adhteb import Benchmark, ModelMetadata

model_name= "my-model-name"
url="https://huggingface.co/my-model-name"

model_metadata = ModelMetadata(model_name=model_name, url=url)
benchmark.publish(model_metadata=model_metadata)
```

## Private Cohorts

As some of the cohort metadata presented in this benchmark is not available publicly, we do by default encrypt private metadata. We are working on finding and extending our benchmark with open, publicly available data as well, which is being used by the benchmark by default. If you want to include results from the private cohorts as well, you can either:

1. Open a [new issue to benchmark a specific model on private cohort data](https://github.com/SCAI-BIO/ADHTEB/issues/new/choose). We will run the benchmark on the non-public data for you and report the results based on your issue (publicly or privately).
2. Get access to the individual cohorts by the data holders and [contact us](mailto:tim.adams@scai.fraunhofer.de) to get a decryption key for the benchmark. You can then run the private benchmark cohorts along the public ones by using the following flag and providing the key:

```python
from adhteb import Benchmark

benchmark = Benchmark(vectorizer=vectorizer, include_private=True, decryption_key=KEY_STRING)
``` 

# Citation

If you use this work in your research, please cite as:

```bibtex
@article{Adams_A_benchmark_of_2025,
  author  = {Adams, Tim and Salimi, Yasamin and Ay, Mehmet Can and
             Valderrama, Diego Felipe and Jacobs, Marc and
             Fröhlich, Holger},
  title   = {A benchmark of text embedding models for semantic harmonization of Alzheimer's disease cohorts},
  journal = {The Journal of Prevention of Alzheimer's Disease},
  volume  = {13},
  number  = {1},
  year    = {2025},
  doi     = {10.1016/j.tjpad.2025.100420}
}
```

**Reference**

Adams T, Salimi Y, Ay MC, Valderrama DF, Jacobs M, Fröhlich H. *A benchmark of text embedding models for semantic harmonization of Alzheimer's disease cohorts*. **The Journal of Prevention of Alzheimer's Disease**. 2025;13(1). https://doi.org/10.1016/j.tjpad.2025.100420
