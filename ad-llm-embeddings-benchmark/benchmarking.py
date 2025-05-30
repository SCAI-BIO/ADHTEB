import logging
import os
from typing import Dict, List

import pandas as pd
import numpy as np
from vectorizers import Vectorizer


class Benchmark:

    logger = logging.getLogger(__name__)

    def __init__(self,
                 vectorizer: Vectorizer,
                 cdm_source: str = "data/AD_CDM_JPAD.csv",
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
        self.vectorizer = vectorizer
        cdm = pd.read_csv(cdm_source, na_values=[""])
        self.groundtruth = self._compute_groundtruth_vectors(cdm)
        self.debug = debug
        self.debug_dest_dir = debug_dest_dir

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

        cohort_with_vectors = cohort.copy()
        cohort_with_vectors["vector"] = cohort_with_vectors["Description"].apply(self.vectorizer.get_embedding)

        cdm_matrix = np.vstack(self.groundtruth["vector"].values)
        cdm_norms = np.linalg.norm(cdm_matrix, axis=1)

        for _, row in cohort_with_vectors.iterrows():
            # if there is no corrsponding vector in the groundtruth, skip this row
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
                    #"groundtruth_definition": self.groundtruth.loc[row["Column_Name"]]["Definition"],
                    "matched_top_1_cdm_variable": self.groundtruth.iloc[top_indices[0]]["Feature"],
                    "matched_top_1_similarity": similarities[top_indices[0]],
                    **{f"in_top_{i + 1}": row["Column_Name"] in top_concepts[:i + 1] for i in range(n)}
                })

        cumulative_accuracies = (correct_at / total).tolist()

        if self.debug:
            os.makedirs(self.debug_dest_dir, exist_ok=True)
            pd.DataFrame(matching_info).to_csv(f"{self.debug_dest_dir}/{self.vectorizer.model_name}_{cohort_name}_matching_info.csv", index=False)

        return cumulative_accuracies

    def get_accuracies(self, n: int = 20) -> Dict[str, List[float]]:
        """
        Get cumulative top-N accuracy lists for all cohorts.

        :param n: Number of top matches to consider.
        :return: Dictionary of cohort name → list of cumulative top-n accuracies.
        """
        results = {}
        cohort_labels = ["GERAS-I", "GERAS-US", "GERAS-J", "GERAS-II", "PREVENT Dementia"]
        cohort_filenames = ["GERAS_I_dict.csv", "GERAS_US_dict.csv", "GERAS_J_dict.csv", "GERAS_II_dict.csv",
                            "PREVENT_DEMENTIA_dict.csv"]

        for cohort_label, cohort_filename in zip(cohort_labels, cohort_filenames):
            cohort = pd.read_csv(f"data/{cohort_filename}")
            # FIXME: we have no definitions in the CDM for these rowsc in PREVENT Dementia -> skip them for now
            if cohort_label == "PREVENT Dementia":
                rows_to_drop = ["medthyrp_act", "medthyrm", "Left_Hippocampus", "Right_Hippocampus", "smoker",
                                "smokern", "smokere"]
                cohort = cohort[~cohort["Column_Name"].isin(rows_to_drop)]

            cumulative_accuracies = self._get_accuracy(cohort, cohort_label, n)
            results[cohort_label] = cumulative_accuracies

        return results
