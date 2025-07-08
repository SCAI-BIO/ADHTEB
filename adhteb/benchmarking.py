import logging
import os
from typing import Dict, List

import pandas as pd
import numpy as np

from vectorizers import Vectorizer
from results import BenchmarkResult


class Benchmark:
    logger = logging.getLogger(__name__)

    def __init__(self,
                 vectorizer: Vectorizer,
                 top_n: int = 20,
                 n_bins: int = 100,
                 debug: bool = True,
                 debug_dest_dir: str = "results") -> None:
        """
        Initialize the Benchmark class.

        :param vectorizer: The vectorizer to be benchmarked.
        :param cdm_source: Path to the Common Data Model (CDM) CSV file.
        :param debug: If True, enables debug mode, will write computed mappings and ground truth to target directory.
        Default is True.
        :param debug_dest_dir: Directory to save debug files. Default is "results".
        """
        # result configuration
        self.top_n = top_n
        self.n_bins = n_bins
        # debugging configuration
        self.debug = debug
        self.debug_dest_dir = debug_dest_dir
        # tested text embedder
        self.vectorizer = vectorizer
        # common data model
        cdm = pd.read_csv("data/AD_CDM_JPAD.csv", na_values=[""])
        self.groundtruth = self._compute_groundtruth_vectors(cdm)
        # GERAS cohorts
        self.geras_i = self._compute_cohort_vectors("data/GERAS_I_dict.csv")
        self.geras_ii = self._compute_cohort_vectors("data/GERAS_II_dict.csv")
        self.geras_us = self._compute_cohort_vectors("data/GERAS_US_dict.csv")
        self.geras_j = self._compute_cohort_vectors("data/GERAS_J_dict.csv")
        # other cohorts
        self.prevent_dementia = self._compute_cohort_vectors("data/PREVENT_DEMENTIA_dict.csv")
        #self.a4 = self._compute_cohort_vectors("data/A4_dict.csv")
        self.aibl = self._compute_cohort_vectors("data/AIBL_dict.csv")
        # Result sets
        self.results_prevent_dementia: BenchmarkResult = None  # n=37
        self.results_geras: BenchmarkResult = None  # n=61
        self.results_aibl: BenchmarkResult = None  # n=54
        #self.results_a4: BenchmarkResult = None  # n=73

    def run(self) -> None:
        """
        Generate results sets for each cohort dataset for configured vectorizer.
        """
        self.logger.info("Benchmarking GERAS cohorts...")
        self.results_geras = self._benchmark_geras()
        self.logger.info("Benchmarking PREVENT Dementia cohort...")
        self.results_prevent_dementia = self._benchmark_cohort(
            self.prevent_dementia, "PREVENT Dementia", self.n_bins)
        self.logger.info("Benchmarking AIBL cohort...")
        self.results_aibl = self._benchmark_cohort(self.aibl, "AIBL", self.n_bins)
        self.logger.info("Benchmarking A4 cohort...")
        #self.results_a4 = self._benchmark_cohort(self.a4, "A4", self.n_bins)
        self.logger.info("Benchmarking completed for all cohorts.")

    def publish(self) -> None:
        """
        Publish benchmark results to leaderboard.
        """
        model_metadata = {
            "name": self.vectorizer.model_name,
            "description": self.vectorizer.description,
            "url": self.vectorizer.url
        }

    def _benchmark_geras(self) -> BenchmarkResult:
        """
        Compute and combine benchmark results from all GERAS cohorts.
        """
        cohort_label = "GERAS"
        n_variables = [len(self.geras_i), len(self.geras_ii),
                       len(self.geras_us), len(self.geras_j), len(self.prevent_dementia)]
        cohorts = [self.geras_i, self.geras_ii, self.geras_us, self.geras_j, self.prevent_dementia]
        cohort_labels = ["GERAS-I", "GERAS-II", "GERAS-US", "GERAS-J", "PREVENT Dementia"]

        total_tp = []
        total_fp = []
        total_fn = []

        zero_shot_accuracies = []

        for cohort, label in zip(cohorts, cohort_labels):
            zero_shot_accuracy = self._get_accuracy(cohort, label, 1)[0]
            zero_shot_accuracies.append(zero_shot_accuracy)

            tp, fp, fn = self._compute_confusion_matrix(cohort, label, self.n_bins)

            total_tp = tp if not total_tp else [x + y for x, y in zip(total_tp, tp)]
            total_fp = fp if not total_fp else [x + y for x, y in zip(total_fp, fp)]
            total_fn = fn if not total_fn else [x + y for x, y in zip(total_fn, fn)]

        # calculate weighted averages (weighted by number of variables) of metrics for all cohorts
        n_total = sum(n_variables)
        precisions = [tp / (tp + fp) if (tp + fp) > 0 else 1.0 for tp, fp in zip(total_tp, total_fp)]
        recalls = [tp / (tp + fn) if (tp + fn) > 0 else 0.0 for tp, fn in zip(total_tp, total_fn)]
        zero_shot_accuracy = sum(zs * n for zs, n in zip(zero_shot_accuracies, n_variables)) / n_total

        # create result set
        results = BenchmarkResult(
            cohort_label=cohort_label,
            n_variables=n_total,
            top_n_accuracy=[zero_shot_accuracy],
            precisions=precisions,
            recalls=recalls
        )

        return results

    def _benchmark_cohort(self, cohort: pd.DataFrame, cohort_name: str, n_bins: int = 100) -> BenchmarkResult:
        """
        Compute benchmark result for a specific cohort.
        
        :param cohort: The cohort DataFrame containing vectors.
        :param cohort_name: Name of the cohort column in CDM.
        :param n_bins: Binning param to calculate similarity thresholds.
        :return: BenchmarkResult containing precision, recall, and zero-shot accuracy.
        """
        self.logger.info(f"Benchmarking cohort {cohort_name}...")

        # compute confusion matrix
        tp, fp, fn = self._compute_confusion_matrix(cohort, cohort_name, n_bins)

        # calculate precision and recall
        precisions = [t / (t + f) if (t + f) > 0 else 1.0 for t, f in zip(tp, fp)]
        recalls = [t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fn)]

        # compute zero-shot accuracy
        zero_shot_accuracy = self._get_accuracy(cohort, cohort_name, 1)[0]

        # create result set
        results = BenchmarkResult(
            cohort_label=cohort_name,
            n_variables=len(cohort),
            top_n_accuracy=[zero_shot_accuracy],
            precisions=precisions,
            recalls=recalls
        )

        return results

    def _compute_confusion_matrix(self, cohort: pd.DataFrame, cohort_name: str, n_bins: int = 100):
        """
        Computes precision and recall for a given cohort DataFrame.
        
        :param cohort: The cohort DataFrame containing vectors.
        :param cohort_name: Name of the cohort column in CDM.
        :param n_bins: Number of thresholds to evaluate between max and min similarity.
        :return: number of TP, FP, FN
        """
        # initialize lists to store results
        true_positives = []
        false_positives = []
        false_negatives = []

        for i in range(n_bins):
            # all vectors with similarity of at least this are considered positives
            min_similarity = 1 - ((1 / n_bins) * i)

            tp = 0
            fp = 0
            fn = 0

            cdm_matrix = np.vstack(self.groundtruth["vector"].values)
            cdm_norms = np.linalg.norm(cdm_matrix, axis=1)

            for _, row in cohort.iterrows():
                # if there is no corresponding vector in the groundtruth, skip this row
                if self.groundtruth[self.groundtruth[cohort_name] == row["Column_Name"]].empty:
                    self.logger.error(f"Skipping row {row['Column_Name']} in cohort {cohort_name} "
                                      f"due to missing ground truth vector.")
                    continue
                vector = row["vector"]
                v_norm = np.linalg.norm(vector)
                similarities = (cdm_matrix @ vector) / (cdm_norms * v_norm)

                # get indices of all vectors that are at least min_similarity
                positive_indices = np.where(similarities >= min_similarity)[0]

                # check if the current variable is in the positive indices
                predicted_labels = self.groundtruth[cohort_name].values[positive_indices]
                actual_label = row["Column_Name"]

                # now: 
                # - all labels that match the current variable count as true positives, can be more than one for upper
                # level concepts
                # - all labels that do not match the current variable count as false positives
                # - if the current variable is not in the positive indices, it counts as a false negative
                if actual_label in predicted_labels:
                    tp += np.sum(predicted_labels == actual_label)
                    fp += np.sum(predicted_labels != actual_label)
                else:
                    fn += 1

            # append for current bin
            true_positives.append(tp)
            false_positives.append(fp)
            false_negatives.append(fn)

        return true_positives, false_positives, false_negatives

    def _compute_groundtruth_vectors(self, cdm: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ground truth vectors for the given Common Data Model (CDM) DataFrame.

        This method reads a predefined CSV file containing the Common Data Model (CDM),
        iterates through its rows, and generates embedding vectors for the "Definition"
        column using the vectorizer. The resulting DataFrame includes the original data
        along with the computed vectors.

        """
        self.logger.info("Computing ground truth vectors...")
        cdm_with_vectors = cdm.copy()
        # drop columns with nan value in "Definition"
        cdm_with_vectors = cdm_with_vectors.dropna(subset=["Definition"], ignore_index=True)
        cdm_with_vectors["vector"] = None
        for idx, row in cdm_with_vectors.iterrows():
            try:
                description = row["Definition"]
                vector = self.vectorizer.get_embedding(description)
                cdm_with_vectors.at[idx, "vector"] = vector
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        return cdm_with_vectors

    def _compute_cohort_vectors(self, cohort_file: str) -> pd.DataFrame:
        """
        Computes vectors for a given cohort DataFrame.

        This method reads a cohort CSV file, iterates through its rows, and generates
        embedding vectors for the "Description" column using the vectorizer. The resulting
        DataFrame includes the original data along with the computed vectors.

        :param cohort_file: Path to the cohort CSV file.
        :return: DataFrame with vectors for each row in the cohort.
        """
        self.logger.info(f"Computing vectors for cohort from {cohort_file}...")
        cohort = pd.read_csv(cohort_file)
        # FIXME: we have no definitions in the CDM for these rows in PREVENT Dementia -> skip them for now
        if cohort_file == "data/PREVENT_DEMENTIA_dict.csv":
            rows_to_drop = ["medthyrp_act", "medthyrm", "Left_Hippocampus", "Right_Hippocampus", "smoker",
                            "smokern", "smokere"]
            cohort = cohort[~cohort["Column_Name"].isin(rows_to_drop)]
        cohort_with_vectors = cohort.copy()
        cohort_with_vectors["vector"] = cohort_with_vectors["Description"].apply(self.vectorizer.get_embedding)
        return cohort_with_vectors

    def _get_accuracy(self, cohort: pd.DataFrame, cohort_name: str, n: int) -> List[float]:
        """
        Compute cumulative top-N accuracy list for a cohort.

        :param cohort: The cohort DataFrame.
        :param cohort_name: Name of the cohort column in CDM.
        :param n: Number of top matches to consider.
        :return: List of length n where the i-th element is the accuracy at top-(i+1).
        """
        self.logger.info(f"Computing top-{n} cumulative accuracy for {cohort_name}...")

        total = len(cohort)
        correct_at = np.zeros(n, dtype=int)
        matching_info = []

        cdm_matrix = np.vstack(self.groundtruth["vector"].values)
        cdm_norms = np.linalg.norm(cdm_matrix, axis=1)

        for _, row in cohort.iterrows():
            # if there is no corresponding vector in the groundtruth, skip this row
            if self.groundtruth[self.groundtruth[cohort_name] == row["Column_Name"]].empty:
                total = total - 1
                continue
            vector = row["vector"]
            v_norm = np.linalg.norm(vector)
            similarities = (cdm_matrix @ vector) / (cdm_norms * v_norm)
            top_indices = np.argsort(similarities)[::-1][:n]
            top_concepts = self.groundtruth.iloc[top_indices][cohort_name].values

            for i in range(n):
                if row["Column_Name"] in top_concepts[:i + 1]:
                    correct_at[i:] += 1
                    break

            if self.debug:
                matching_info.append({
                    "cohort_variable": row["Column_Name"],
                    "cohort_definition": row["Description"],
                    "matched_top_1_cdm_definition": self.groundtruth.iloc[top_indices[0]]["Definition"],
                    # "groundtruth_definition": self.groundtruth.loc[row["Column_Name"]]["Definition"],
                    "matched_top_1_cdm_variable": self.groundtruth.iloc[top_indices[0]]["Feature"],
                    "matched_top_1_similarity": similarities[top_indices[0]],
                    **{f"in_top_{i + 1}": row["Column_Name"] in top_concepts[:i + 1] for i in range(n)}
                })

        cumulative_accuracies = (correct_at / total).tolist()

        if self.debug:
            os.makedirs(self.debug_dest_dir, exist_ok=True)
            pd.DataFrame(matching_info).to_csv(
                f"{self.debug_dest_dir}/{self.vectorizer.model_name}_{cohort_name}_matching_info.csv", index=False)

        return cumulative_accuracies
