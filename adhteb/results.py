from typing import List
from pydantic import BaseModel, computed_field
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



    @computed_field(return_type=float)
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

    def __str__(self) -> str:
        top_n_acc = ', '.join(f'{acc: .2f}' for acc in self.top_n_accuracy)
        return (
            f"BenchmarkResult(cohort='{self.cohort_label}', "
            f"n_variables={self.n_variables}, "
            f"top_n_accuracy=[{top_n_acc}], "
            f"auc={self.auc: .4f})"
        )

