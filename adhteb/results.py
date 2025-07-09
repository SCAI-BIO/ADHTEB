from typing import List

from matplotlib import pyplot as plt
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

    model_config = {
        "frozen": True
    }

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

    def save_pr_curve(self, output_dir: str = "plots") -> str:
        """
        Save the Precision-Recall curve plot to a PNG file.

        :param output_dir: Directory to save the plot.
        :return: Path to the saved plot file.
        """
        import os
        import matplotlib.pyplot as plt

        if not self.precisions or not self.recalls:
            print(f"Skipping PR curve for {self.cohort_label}: missing precision or recall.")
            return ""

        # Sort recalls and precisions if necessary
        if self.recalls[0] > self.recalls[-1]:
            recalls = self.recalls[::-1]
            precisions = self.precisions[::-1]
        else:
            recalls = self.recalls
            precisions = self.precisions

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set up plot
        fig, ax = plt.subplots()
        ax.step(recalls, precisions, where='post', label=f'AUPRC = {self.auc:.4f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve: {self.cohort_label}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend()
        ax.grid(True)

        # Save plot
        filename = f"pr_curve_{self.cohort_label.replace(' ', '_').lower()}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        plt.close(fig)  # Important to avoid memory leaks in batch runs

        print(f"Saved PR curve to {path}")
        return path

    def __str__(self) -> str:
        top_n_acc = ', '.join(f'{acc: .2f}' for acc in self.top_n_accuracy)
        return (
            f"BenchmarkResult(cohort='{self.cohort_label}', "
            f"n_variables={self.n_variables}, "
            f"top_n_accuracy=[{top_n_acc}], "
            f"auc={self.auc: .4f})"
        )

