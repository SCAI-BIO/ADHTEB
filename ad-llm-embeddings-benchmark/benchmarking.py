import logging
import os
from typing import Dict, List
from warnings import deprecated

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
        self.vectorizer  = vectorizer
        # common data model
        cdm = pd.read_csv("data/AD_CDM_JPAD.csv", na_values=[""])
        self.groundtruth = self._compute_groundtruth_vectors(cdm)
        # GERAS cohorts
        self.geras_i = self._compute_cohort_vectors("data/GERAS_I_dict.csv")
        self.geras_ii = self._compute_cohort_vectors("data/GERAS_II_dict.csv")
        self.geras_us = self._compute_cohort_vectors("data/GERAS_US_dict.csv")
        self.geras_j = self._compute_cohort_vectors("data/GERAS_J_dict.csv")
        self.prevent_dementia = self._compute_cohort_vectors("data/PREVENT_DEMENTIA_dict.csv")
        # other cohorts 
        # TODO
        # Result sets
        results_geras: BenchmarkResult = None
        results_adni: BenchmarkResult = None
        results_abvib: BenchmarkResult = None
        results_aibl: BenchmarkResult = None
        results_a4: BenchmarkResult = None
        
    
    def run(self) -> None:
        """
        Generate results sets for each cohort dataset for configured vectorizer.
        """
        self.logger.info("Benchmarking GERAS cohorts...")
        self.results_geras = self._benchmark_geras()
        self.logger.info("Benchmarking completed.")


    def publish(self) -> None:
        """
        Publish benchmark results 
        """


    def _benchmark_geras(self) -> BenchmarkResult:
        """
        Compute and combine benchmark results from all GERAS cohorts + PREVENT Dementia.
        """
        cohort_label = "GERAS"
        n_variables = len(self.geras_i) + len(self.geras_ii) + \
                       len(self.geras_us) + len(self.geras_j) + len(self.prevent_dementia)  
        top_n_accuracy = self.get_accuracies(n=20)  
        return None
        


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

    @deprecated
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

    @deprecated
    def get_precision_recall(self, n_bins: int = 100) -> Dict[str, Dict[str, List[float]]]:
        """
        Get precision and recall curves for each cohort over a fixed range of
        similarity thresholds.

        :param n_bins: Number of thresholds to evaluate between max and min similarity.
        :return: dict mapping cohort name to {'precision': [...], 'recall': [...]}.
        """
        results: Dict[str, Dict[str, List[float]]] = {}
        cohort_labels = ["GERAS-I", "GERAS-US", "GERAS-J", "GERAS-II", "PREVENT Dementia"]
        cohorts = [self.geras_i, self.geras_us, self.geras_j, self.geras_ii, self.prevent_dementia]

        # precompute CDM embeddings matrix once
        cdm_matrix = np.vstack(self.groundtruth["vector"].values)
        cdm_norms = np.linalg.norm(cdm_matrix, axis=1)

        for cohort_label, cohort in zip(cohort_labels, cohorts):
            sims = []
            gt_indices_list = []

            for _, row in cohort.iterrows():
                # compute similarities of this cohort vector to all CDM vectors
                v = row["vector"]
                v_norm = np.linalg.norm(v)
                sim = (cdm_matrix @ v) / (cdm_norms * v_norm)
                sims.append(sim)

                # find _all_ indices in groundtruth that match this cohort variable
                matches = np.where(self.groundtruth[cohort_label] == row["Column_Name"])[0]
                if matches.size == 0:
                    # no groundtruth: skip this row entirely
                    continue
                gt_indices_list.append(matches.tolist())

            if not gt_indices_list:
                raise ValueError(f"No groundtruth found for any row in cohort {cohort_label}")

            sims = np.array(sims)  # shape = (n_cohort, n_cdm)
            n_cohort, n_cdm = sims.shape

            # flatten sims and build binary labels: 1 if (i,j) is any true match
            sims_flat = sims.ravel()
            labels = np.zeros_like(sims_flat, dtype=int)

            total_positives = 0
            for i, match_indices in enumerate(gt_indices_list):
                total_positives += len(match_indices)
                base = i * n_cdm
                for idx in match_indices:
                    labels[base + idx] = 1

            # define thresholds between max and min similarity
            max_sim, min_sim = sims_flat.max(), sims_flat.min()
            thresholds = np.linspace(max_sim, min_sim, n_bins)

            precision = []
            recall = []

            for t in thresholds:
                preds = sims_flat >= t
                TP = int((preds & (labels == 1)).sum())
                FP = int((preds & (labels == 0)).sum())
                FN = total_positives - TP

                prec = TP / (TP + FP) if (TP + FP) > 0 else 1.0
                rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                precision.append(prec)
                recall.append(rec)

            results[cohort_label] = {
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds.tolist()
            }

        return results

    @deprecated
    def get_accuracies(self, n: int = 20) -> Dict[str, List[float]]:
        """
        Get cumulative top-N accuracy lists for all cohorts.

        :param n: Number of top matches to consider.
        :return: Dictionary of cohort name → list of cumulative top-n accuracies.
        """
        results = {}
        cohort_labels = ["GERAS-I", "GERAS-US", "GERAS-J", "GERAS-II", "PREVENT Dementia"]
        cohorts = [self.geras_i, self.geras_us, self.geras_j, self.geras_ii, self.prevent_dementia]

        for cohort_label, cohort in zip(cohort_labels, cohorts):
            cumulative_accuracies = self._get_accuracy(cohort, cohort_label, n)
            results[cohort_label] = cumulative_accuracies

        return results
