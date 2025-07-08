from typing import List
from pydantic import BaseModel
from sklearn.metrics import auc

class BenchmarkResult(BaseModel):
    """
    BenchmarkResult for a specific (cohort) dataset.
    """
    cohort_label: str
    n_variables: int
    top_n_accuracy: List[float]
    precisions: List[float]
    recalls: List[float]

    @property
    def auc(self) -> float:
        """
        Area under the precision-recall curve.
        Computed on the fly from `precisions` and `recalls`.
        """
        if not self.precisions or not self.recalls:
            return 0.0

        # Ensure recall is sorted
        if self.recalls[0] > self.recalls[-1]:
            recalls = self.recalls[::-1]
            precisions = self.precisions[::-1]
        else:
            recalls = self.recalls
            precisions = self.precisions

        return auc(recalls, precisions)

    class Config:
        allow_mutation = False
