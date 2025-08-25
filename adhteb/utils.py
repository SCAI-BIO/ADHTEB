import os
from typing import List

from adhteb.results import BenchmarkResult


def fix_blas_float_variability() -> None:
    """
    Reduces BLAS-threading-induced floating point variability.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def aggregate_score(results: List[BenchmarkResult]) -> float:
    """
    Computes an aggregated score for all cohorts based on their AUC values and zero-shot accuracies, weighted by the
    number of variables per cohort.

    :return: Composite score as a float.
    """
    if results is None or len(results) == 0:
        raise ValueError("Benchmark results are empty. Please run the benchmark first.")

    total_score = 0.0
    total_n_variables = 0

    for result in results:
        auc = result.auc
        n_variables = result.n_variables
        zero_shot_accuracy = result.top_n_accuracy[0]
        score = ((0.5 * auc) + (0.5 * zero_shot_accuracy)) * n_variables
        total_score += score
        total_n_variables += n_variables

    return total_score / total_n_variables
