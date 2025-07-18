import importlib
import logging
import os
import pickle
import copy
from io import StringIO

import pandas as pd
import numpy as np
import importlib.resources as pkg_resources

from typing import List

from cryptography.fernet import Fernet
from tabulate import tabulate

from .leaderboard import LeaderboardEntry, ModelMetadata, publish_entry
from .vectorizers import Vectorizer, GeminiVectorizer, OpenAIVectorizer  # Import specific vectorizers for batching
from .results import BenchmarkResult


class Benchmark:
    logger = logging.getLogger(__name__)

    def __init__(self,
                 vectorizer: Vectorizer,
                 top_n: int = 20,
                 n_bins: int = 100,
                 debug: bool = False,
                 debug_dest_dir: str = "results") -> None:
        """
        Initialize the Benchmark class.

        :param vectorizer: The vectorizer to be benchmarked.
        :param top_n: Number of top matches to consider for accuracy calculation.
        Default is 20.
        :param n_bins: Number of bins to use for similarity thresholds in confusion matrix calculation.
        :param debug: If True, enables debug mode, will write computed mappings and ground truth to target directory.
        Default is False.
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
        cdm = self.__load_data("AD_CDM_JPAD.csv", na_values=[""])
        self.__groundtruth = self._compute_groundtruth_vectors(cdm)
        # GERAS cohorts
        self.__geras_i = self._compute_cohort_vectors(self.__load_data("GERAS_I_dict.csv"))
        self.__geras_ii = self._compute_cohort_vectors(self.__load_data("GERAS_II_dict.csv"))
        self.__geras_us = self._compute_cohort_vectors(self.__load_data("GERAS_US_dict.csv"))
        self.__geras_j = self._compute_cohort_vectors(self.__load_data("GERAS_J_dict.csv"))
        # other cohorts
        self.__prevent_dementia = self._compute_cohort_vectors(self.__load_data("PREVENT_DEMENTIA_dict.csv"))
        self.__prevent_ad = self._compute_cohort_vectors(self.__load_data("PREVENT_AD_dict.csv"))
        self.__emif = self._compute_cohort_vectors(self.__load_data("EMIF_dict.csv"))
        # Result sets
        self.results_prevent_dementia: BenchmarkResult = None
        self.results_geras: BenchmarkResult = None
        self.results_prevent_ad: BenchmarkResult = None
        self.results_emif: BenchmarkResult = None

    def __load_key(self):
        with importlib.resources.path('adhteb.data', 'key.bin') as data_path:
            with open(data_path, 'rb') as f:
                key = f.read()
        return key

    def __load_data(self, file_name: str, **read_csv_kwargs) -> pd.DataFrame:
        """
        Load and decrypt an encrypted CSV file from the package, returning it as a DataFrame.
        Decryption is performed entirely in memory using Fernet.
        """
        fernet = Fernet(self.__load_key())
        data_path = pkg_resources.files('adhteb.data').joinpath(file_name)
        with data_path.open('rb') as encrypted_file:
            encrypted_data = encrypted_file.read()
            decrypted_data = fernet.decrypt(encrypted_data)
        return pd.read_csv(StringIO(decrypted_data.decode('utf-8')), **read_csv_kwargs)

    def run(self) -> None:
        """
        Generate results sets for each cohort dataset for configured vectorizer.
        """
        self.logger.info("Benchmarking GERAS cohorts...")
        self.__geras_i = self._drop_cohort_records_without_groundtruth(self.__geras_i, "GERAS-I")
        self.__geras_ii = self._drop_cohort_records_without_groundtruth(self.__geras_ii, "GERAS-II")
        self.__geras_us = self._drop_cohort_records_without_groundtruth(self.__geras_us, "GERAS-US")
        self.__geras_j = self._drop_cohort_records_without_groundtruth(self.__geras_j, "GERAS-J")
        self.results_geras = self._benchmark_geras()

        self.logger.info("Benchmarking PREVENT Dementia cohort...")
        self.__prevent_dementia = self._drop_cohort_records_without_groundtruth(self.__prevent_dementia,
                                                                                "PREVENT Dementia")
        self.results_prevent_dementia = self._benchmark_cohort(self.__prevent_dementia, "PREVENT Dementia", self.n_bins)

        self.logger.info("Benchmarking EMIF cohort...")
        self.__emif = self._drop_cohort_records_without_groundtruth(self.__emif, "EMIF")
        self.results_emif = self._benchmark_cohort(self.__emif, "EMIF", self.n_bins)

        self.logger.info("Benchmarking PREVENT-AD cohort...")
        self.__prevent_ad = self._drop_cohort_records_without_groundtruth(self.__prevent_ad, "PREVENT-AD")
        self.results_prevent_ad = self._benchmark_cohort(self.__prevent_ad, "PREVENT-AD", self.n_bins)
        self.logger.info("Benchmarking completed for all cohorts.")

    def save(self, path):
        copy_self = copy.copy(self)
        # can't pickle this due to threading
        copy_self.vectorizer = None

        with open(path, 'wb') as f:
            pickle.dump(copy_self, f)

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
            self.results_prevent_ad,
            self.results_emif
        ]):
            raise ValueError("Benchmark results for all cohorts must be computed before generating summary.")

        summary_data = {
            "GERAS": [self.results_geras.auc, self.results_geras.top_n_accuracy[0]],
            "PREVENT Dementia": [self.results_prevent_dementia.auc, self.results_prevent_dementia.top_n_accuracy[0]],
            "PREVENT AD": [self.results_prevent_ad.auc, self.results_prevent_ad.top_n_accuracy[0]],
            "EMIF": [self.results_emif.auc, self.results_emif.top_n_accuracy[0]],
        }

        summary_df = pd.DataFrame(summary_data, index=["AUPRC", "Zero-shot Accuracy"]).T
        summary_df = summary_df.round(2)

        aggregate_score = self.aggregate_score()

        return f"{tabulate(summary_df, headers='keys', tablefmt='pretty')}\nAggregate Score: {aggregate_score:.2f}"

    def aggregate_score(self) -> float:
        """
        Computes an aggregated score for all cohorts based on their AUC values and zero-shot accuracies, weighted by the
        number of variables per cohort.

        :return: Composite score as a float.
        """
        if not all([self.results_geras, self.results_prevent_dementia, self.results_prevent_ad, self.results_emif]):
            raise ValueError("Benchmark results for all cohorts must be computed before aggregating score.")

        total_score = 0.0
        total_n_variables = 0

        for results in [self.results_geras, self.results_prevent_dementia, self.results_prevent_ad, self.results_emif]:
            auc = results.auc
            n_variables = results.n_variables
            zero_shot_accuracy = results.top_n_accuracy[0]
            score = ((0.5 * auc) + (0.5 * zero_shot_accuracy)) * n_variables
            total_score += score
            total_n_variables += n_variables

        return total_score / total_n_variables

    def publish(self, metadata: ModelMetadata) -> None:
        """
        Publish benchmark results to leaderboard.
        """
        model_metadata = metadata
        entry = LeaderboardEntry(
            model=model_metadata,
            aggregate_score=self.aggregate_score(),
            cohort_benchmarks=[
                BenchmarkResult(**self.results_geras.model_dump()),
                BenchmarkResult(**self.results_prevent_dementia.model_dump()),
                BenchmarkResult(**self.results_prevent_ad.model_dump()),
                BenchmarkResult(**self.results_emif.model_dump()),
            ]
        )
        print(entry.model_dump_json())
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
        valid_rows = cohort[cohort["Column_Name"].isin(self.__groundtruth[cohort_name])]
        if len(valid_rows) < len(cohort):
            self.logger.warning(f"Dropped {len(cohort) - len(valid_rows)} records from cohort {cohort_name} "
                                f"due to missing ground truth vectors.")
            dropped_rows = cohort[~cohort["Column_Name"].isin(self.__groundtruth[cohort_name])]
            self.logger.debug(f'The dropped records were: {dropped_rows}')
        return valid_rows

    def _benchmark_geras(self) -> BenchmarkResult:
        """
        Compute and combine benchmark results from all GERAS cohorts.
        """
        cohort_label = "GERAS"
        n_variables = [len(self.__geras_i), len(self.__geras_ii),
                       len(self.__geras_us), len(self.__geras_j)]
        cohorts = [self.__geras_i, self.__geras_ii, self.__geras_us, self.__geras_j]
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

            cdm_matrix = np.vstack(self.__groundtruth["vector"].values)
            cdm_norms = np.linalg.norm(cdm_matrix, axis=1)

            for _, row in cohort.iterrows():
                # if there is no corresponding vector in the groundtruth, skip this row
                if self.__groundtruth[self.__groundtruth[cohort_name] == row["Column_Name"]].empty:
                    self.logger.error(f"Skipping row {row['Column_Name']} in cohort {cohort_name} "
                                      f"due to missing ground truth vector.")
                    continue
                vector = row["vector"]
                v_norm = np.linalg.norm(vector)
                similarities = (cdm_matrix @ vector) / (cdm_norms * v_norm)

                # get indices of all vectors that are at least min_similarity
                positive_indices = np.where(similarities >= min_similarity)[0]

                # check if the current variable is in the positive indices
                predicted_labels = self.__groundtruth[cohort_name].values[positive_indices]
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
        cdm_with_vectors = cdm_with_vectors.dropna(subset=["Definition"], ignore_index=True)

        descriptions = cdm_with_vectors["Definition"].tolist()

        # Determine batch size based on the vectorizer type
        if isinstance(self.vectorizer, GeminiVectorizer):
            batch_size = 50  # Gemini embedding batch size limit is 100, but 50 is safer for rate limits
        elif isinstance(self.vectorizer, OpenAIVectorizer):
            batch_size = 200  # OpenAI recommends 2048, but 200 is safer for rate limits
        else:
            batch_size = len(descriptions)  # For other vectorizers, process all at once

        all_vectors = []
        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i + batch_size]
            try:
                # Use get_embeddings_batch if available, otherwise fallback to individual embeddings
                if hasattr(self.vectorizer, 'get_embeddings_batch'):
                    batch_vectors = self.vectorizer.get_embeddings_batch(batch_descriptions)
                else:
                    batch_vectors = [self.vectorizer.get_embedding(desc) for desc in batch_descriptions]
                all_vectors.extend(batch_vectors)
            except Exception as e:
                self.logger.error(f"Error processing batch {i} to {i + batch_size}: {e}")
                # If batch fails, try individual for the failed batch to isolate issues
                for desc in batch_descriptions:
                    try:
                        all_vectors.append(self.vectorizer.get_embedding(desc))
                    except Exception as individual_e:
                        self.logger.error(f"Error processing individual description '{desc}': {individual_e}")
                        all_vectors.append(None)  # Append None if individual fails too

        cdm_with_vectors["vector"] = all_vectors
        # Drop rows where vector calculation failed (if any)
        cdm_with_vectors = cdm_with_vectors.dropna(subset=["vector"]).reset_index(drop=True)
        return cdm_with_vectors

    def _compute_cohort_vectors(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Computes vectors for a given cohort DataFrame.

        This method reads a cohort CSV file, iterates through its rows, and generates
        embedding vectors for the "Description" column using the vectorizer. The resulting
        DataFrame includes the original data along with the computed vectors.

        :param cohort: The cohort DataFrame containing descriptions.
        :return: DataFrame with vectors for each row in the cohort.
        """
        self.logger.info(f"Computing cohort vectors...")
        cohort = self._drop_cohort_records_without_descriptions(cohort)
        cohort_with_vectors = cohort.copy()

        descriptions = cohort_with_vectors["Description"].tolist()

        # Determine batch size based on the vectorizer type
        if isinstance(self.vectorizer, GeminiVectorizer):
            batch_size = 50  # Gemini embedding batch size limit is 100, but 50 is safer for rate limits
        elif isinstance(self.vectorizer, OpenAIVectorizer):
            batch_size = 200  # OpenAI recommends 2048, but 200 is safer for rate limits
        else:
            batch_size = len(descriptions)  # For other vectorizers, process all at once

        all_vectors = []
        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i + batch_size]
            try:
                if hasattr(self.vectorizer, 'get_embeddings_batch'):
                    batch_vectors = self.vectorizer.get_embeddings_batch(batch_descriptions)
                else:
                    batch_vectors = [self.vectorizer.get_embedding(desc) for desc in batch_descriptions]
                all_vectors.extend(batch_vectors)
            except Exception as e:
                self.logger.error(f"Error processing batch {i} to {i + batch_size}: {e}")
                for desc in batch_descriptions:
                    try:
                        all_vectors.append(self.vectorizer.get_embedding(desc))
                    except Exception as individual_e:
                        self.logger.error(f"Error processing individual description '{desc}': {individual_e}")
                        all_vectors.append(None)

        cohort_with_vectors["vector"] = all_vectors
        cohort_with_vectors = cohort_with_vectors.dropna(subset=["vector"]).reset_index(drop=True)
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
            self.logger.debug(f'The dropped rows were: {dropped_rows}')
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

        cdm_matrix = np.vstack(self.__groundtruth["vector"].values)
        cdm_norms = np.linalg.norm(cdm_matrix, axis=1)

        for _, row in cohort.iterrows():
            # if there is no corresponding vector in the groundtruth, skip this row
            if self.__groundtruth[self.__groundtruth[cohort_name] == row["Column_Name"]].empty:
                total = total - 1
                continue
            vector = row["vector"]
            v_norm = np.linalg.norm(vector)
            similarities = (cdm_matrix @ vector) / (cdm_norms * v_norm)
            top_indices = np.argsort(similarities)[::-1][:n]
            top_concepts = self.__groundtruth.iloc[top_indices][cohort_name].values

            for i in range(n):
                if row["Column_Name"] in top_concepts[:i + 1]:
                    correct_at[i:] += 1
                    break

            if self.debug:
                matching_info.append({
                    "cohort_variable": row["Column_Name"],
                    "cohort_definition": row["Description"],
                    "matched_top_1_cdm_definition": self.__groundtruth.iloc[top_indices[0]]["Definition"],
                    "matched_top_1_cdm_variable": self.__groundtruth.iloc[top_indices[0]]["Feature"],
                    "matched_top_1_similarity": similarities[top_indices[0]],
                    **{f"in_top_{i + 1}": row["Column_Name"] in top_concepts[:i + 1] for i in range(n)}
                })

        cumulative_accuracies = (correct_at / total).tolist()

        if self.debug:
            os.makedirs(self.debug_dest_dir, exist_ok=True)
            pd.DataFrame(matching_info).to_csv(
                f"{self.debug_dest_dir}/{self.vectorizer.model_name}_{cohort_name}_matching_info.csv", index=False)

        return cumulative_accuracies
