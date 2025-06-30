from typing import List
from pydantic import BaseModel

class BenchmarkResult(BaseModel):
    """
    BenchmarkResult for a specific (cohort) datatset.
    """
    cohort_label: str
    n_variables: int
    top_n_accuracy: List[float]
    precisions: List[float]
    recalls: List[float]

    class Config:
        allow_mutation = False