import logging
import os
from typing import List

import pandas as pd
import numpy as np
from tabulate import tabulate

from adhteb.leaderboard import LeaderboardEntry, publish_entry
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
        self.a4 = self._compute_cohort_vectors("data/A4_dict.csv")
        self.aibl = self._compute_cohort_vectors("data/AIBL_dict.csv")
        # Result sets
        self.results_prevent_dementia: BenchmarkResult = None  # n=37
        self.results_geras: BenchmarkResult = None  # n=61
        self.results_aibl: BenchmarkResult = None  # n=54
        self.results_a4: BenchmarkResult = None  # n=73

    def run(self) -> None:
        """
        Generate results sets for each cohort dataset for configured vectorizer.
        """
        self.logger.info("Benchmarking GERAS cohorts...")
        self.geras_i = self._drop_cohort_records_without_groundtruth(self.geras_i, "GERAS-I")
        self.geras_ii = self._drop_cohort_records_without_groundtruth(self.geras_ii, "GERAS-II")
        self.geras_us = self._drop_cohort_records_without_groundtruth(self.geras_us, "GERAS-US")
        self.geras_j = self._drop_cohort_records_without_groundtruth(self.geras_j, "GERAS-J")
        self.results_geras = self._benchmark_geras()
        self.logger.info("Benchmarking PREVENT Dementia cohort...")
        self.prevent_dementia = self._drop_cohort_records_without_groundtruth(self.prevent_dementia, "PREVENT Dementia")
        self.results_prevent_dementia = self._benchmark_cohort(self.prevent_dementia, "PREVENT Dementia", self.n_bins)
        self.logger.info("Benchmarking AIBL cohort...")
        self.aibl = self._drop_cohort_records_without_groundtruth(self.aibl, "AIBL")
        self.results_aibl = self._benchmark_cohort(self.aibl, "AIBL", self.n_bins)
        self.logger.info("Benchmarking A4 cohort...")
        self.a4 = self._drop_cohort_records_without_groundtruth(self.a4, "A4")
        self.results_a4 = self._benchmark_cohort(self.a4, "A4", self.n_bins)
        self.logger.info("Benchmarking completed for all cohorts.")

    def results_summary(self) -> str:
        """
        Generates a summary of the benchmark results for all cohorts.
        Stores AUPRC and zero-shot accuracy in a DataFrame and pretty prints the results.
        Results are rounded to 2 decimal places and the DataFrame is transposed so that
        cohorts are rows and metrics are columns.
        """
        if not all([
            self.results_geras,
            self.results_prevent_dementia,
            self.results_aibl,
            self.results_a4
        ]):
            raise ValueError("Benchmark results for all cohorts must be computed before generating summary.")

        summary_data = {
            "GERAS": [self.results_geras.auc, self.results_geras.top_n_accuracy[0]],
            "PREVENT Dementia": [self.results_prevent_dementia.auc, self.results_prevent_dementia.top_n_accuracy[0]],
            "AIBL": [self.results_aibl.auc, self.results_aibl.top_n_accuracy[0]],
            "A4": [self.results_a4.auc, self.results_a4.top_n_accuracy[0]],
        }

        summary_df = pd.DataFrame(summary_data, index=["AUPRC", "Zero-shot Accuracy"]).T
        summary_df = summary_df.round(2)

        aggregate_score = self.aggregate_score()

        return f"{tabulate(summary_df, headers='keys', tablefmt='pretty')}\nAggregate Score: {aggregate_score:.2f}"

    def aggregate_score(self) -> float:
        """
        Computes a aggregated score for all cohorts based on their AUC values and zero-shot accuracies, weighted by the
        number of variables per cohort.

        :return: Composite score as a float.
        """
        if not all([self.results_geras, self.results_prevent_dementia, self.results_aibl, self.results_a4]):
            raise ValueError("Benchmark results for all cohorts must be computed before aggregating score.")

        total_score = 0.0
        total_n_variables = 0

        for results in [self.results_geras, self.results_prevent_dementia, self.results_aibl, self.results_a4]:
            auc = results.auc
            n_variables = results.n_variables
            zero_shot_accuracy = results.top_n_accuracy[0]
            score = ((0.5 * auc) + (0.5 * zero_shot_accuracy)) * n_variables
            total_score += score
            total_n_variables += n_variables

        return total_score / total_n_variables

    def publish(self) -> None:
        """
        Publish benchmark results to leaderboard.
        """
        model_metadata = {
            "name": self.vectorizer.model_name,
            "url": ""
        }
        entry = LeaderboardEntry(
            model=model_metadata,
            cohort_benchmarks=[
                self.results_geras.model_dump(),
                self.results_prevent_dementia.model_dump(),
                self.results_aibl.model_dump(),
                self.results_a4.model_dump()
            ]
        )
        self.logger.info("Publishing benchmark results to leaderboard...")
        publish_entry(entry)

    def _drop_cohort_records_without_groundtruth(self, cohort: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
        """
        Drop records from the cohort that Column_Name does not exist in the ground truth for that cohort name.

        :param cohort: The cohort DataFrame.
        :param cohort_name: Name of the cohort column in CDM.
        """
        self.logger.debug(f"Dropping records without ground truth for cohort {cohort_name}...")
        # filter out rows where Column_Name does not exist in the ground truth for the given cohort name
        valid_rows = cohort[cohort["Column_Name"].isin(self.groundtruth[cohort_name])]
        if len(valid_rows) < len(cohort):
            self.logger.warning(f"Dropped {len(cohort) - len(valid_rows)} records from cohort {cohort_name} "
                                f"due to missing ground truth vectors.")
            dropped_rows = cohort[~cohort["Column_Name"].isin(self.groundtruth[cohort_name])]
            self.logger.debug(f'The dropped records were: {dropped_rows}')
        return valid_rows

    def _benchmark_geras(self) -> BenchmarkResult:
        """
        Compute and combine benchmark results from all GERAS cohorts.
        """
        cohort_label = "GERAS"
        n_variables = [len(self.geras_i), len(self.geras_ii),
                       len(self.geras_us), len(self.geras_j)]
        cohorts = [self.geras_i, self.geras_ii, self.geras_us, self.geras_j]
        cohort_labels = ["GERAS-I", "GERAS-II", "GERAS-US", "GERAS-J"]

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
        precisions = [tp / (tp + fp) if (tp + fp) > 0 else 1.0 for tp, fp in zip(tp, fp)]
        recalls = [tp / (tp + fn) if (tp + fn) > 0 else 0.0 for tp, fn in zip(tp, fn)]

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
        # drop rows with invalid or empty descriptions
        cohort = self._drop_cohort_records_without_descriptions(cohort)
        cohort_with_vectors = cohort.copy()
        cohort_with_vectors["vector"] = cohort_with_vectors["Description"].apply(self.vectorizer.get_embedding)
        return cohort_with_vectors

    def _drop_cohort_records_without_descriptions(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Drop records from the cohort that do not have a description.

        :param cohort: The cohort DataFrame.
        :return: Filtered DataFrame with only records that have a description.
        """
        self.logger.debug("Dropping records without descriptions...")
        # filter out rows where Description is NaN or empty
        valid_rows = cohort[cohort["Description"].notna() & (cohort["Description"] != "")]
        if len(valid_rows) < len(cohort):
            self.logger.warning(f"Dropped {len(cohort) - len(valid_rows)} records from cohort "
                                f"due to missing descriptions.")
            dropped_rows = cohort[~cohort["Description"].notna() | (cohort["Description"] == "")]
            self.logger.debug(f'The dropped records were: {dropped_rows}')
        return valid_rows

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
