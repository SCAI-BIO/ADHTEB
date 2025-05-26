import logging
import os

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

    def _get_accuracy(self, cohort: pd.DataFrame, cohort_name: str) -> float:
        """
        Compute the accuracy of the vectorizer on a given cohort.

        :param cohort: The cohort DataFrame containing the definitions to be matched.
        :param cohort_name: The name of the cohort column in the CDM
        """

        self.logger.info(f"Computing accuracy for {cohort_name}...")

        num_definitions_total = len(cohort)
        num_definitions_correct = 0

        # hold matching information
        matching_info = []

        # compute embeddings for the cohort
        cohort_with_vectors = cohort.copy()
        cohort_with_vectors["vector"] = None
        for idx, row in cohort.iterrows():
            description = row["Description"]
            vector = self.vectorizer.get_embedding(description)
            cohort_with_vectors.at[idx, "vector"] = vector

        cdm_matrix = np.vstack(self.groundtruth["vector"].values)
        cdm_norms = np.linalg.norm(cdm_matrix, axis=1)

        for idx, row in cohort_with_vectors.iterrows():
            vector = row["vector"]
            # compute once per cohort vector
            v_norm = np.linalg.norm(vector)
            similarities = (cdm_matrix @ vector) / (cdm_norms * v_norm)
            closest_idx = np.argmax(similarities)
            similarity = similarities[closest_idx]
            closest_description = self.groundtruth.at[closest_idx, "Definition"]
            matched_cdm_concept = self.groundtruth.at[closest_idx, cohort_name]
            if matched_cdm_concept == row["Column_Name"]:
                num_definitions_correct += 1
                matched_correctly = True
            else:
                matched_correctly = False
            if self.debug:
                matching_info.append({
                    "cohort_variable": row["Column_Name"],
                    "cohort_definition": row["Description"],
                    "matched_cdm_definition": closest_description,
                    "matched_cdm_concept": matched_cdm_concept,
                    "similarity": similarity,
                    "matched_correctly": matched_correctly
                })
        # compute accuracy
        accuracy = num_definitions_correct / num_definitions_total
        if self.debug:
            # save matching information to CSV
            matching_info_df = pd.DataFrame(matching_info)
            os.makedirs(self.debug_dest_dir, exist_ok=True)
            matching_info_df.to_csv(f"{self.debug_dest_dir}/{cohort_name}_matching_info.csv", index=False)
        return accuracy

    def get_accuracies(self):
        """
        Get the accuracies of the vectorizer.
        """
        results = {}
        # contains the correct mappings
        groundtruth = self.groundtruth
        # column headers for cohorts in CDM
        cohort_labels = ["GERAS-I", "GERAS-US", "GERAS-J", "GERAS-II", "PREVENT Dementia"]
        # "PREVENT_DEMENTIA_dict.csv" skipped for now
        cohort_filenames = ["GERAS_I_dict.csv", "GERAS_US_dict.csv", "GERAS_J_dict.csv", "GERAS_II_dict.csv",
                            "PREVENT_DEMENTIA_dict.csv"]
        # compute accuracies for each cohort
        for cohort_label, cohort_filename in zip(cohort_labels, cohort_filenames):
            # read the cohort file
            cohort = pd.read_csv(f"data/{cohort_filename}")
            # FIXME: we have no definitions in the CDM for these rowsc in PREVENT Dementia -> skip them for now
            if cohort_label == "PREVENT Dementia":
                rows_to_drop = ["medthyrp_act", "medthyrm", "Left_Hippocampus", "Right_Hippocampus", "smoker",
                                "smokern", "smokere"]
                # drop all rows where cohort["Column_Name"] is in rows_to_drop
                cohort = cohort[~cohort["Column_Name"].isin(rows_to_drop)]
            # compute the accuracy
            accuracy = self._get_accuracy(cohort, cohort_label)
            # store the accuracy
            results[cohort_label] = accuracy
        return results
