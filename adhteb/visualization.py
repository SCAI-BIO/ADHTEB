import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from adhteb import BenchmarkResult, Benchmark  # Assuming correct import path


def plot_prauc_curves_per_cohort(
        benchmarks: List[Benchmark],
        labels: List[str],
        output_dir: str = "plots"
) -> None:
    """
    Generates and saves Precision-Recall (PR) curves for each cohort, comparing
    different vectorizers. Each plot will show the PR curves for all
    benchmarked models for a specific cohort.

    Applies Seaborn styling, uses a custom color scheme, and sorts legend
    labels by AUPRC in descending order.

    :param benchmarks: A list of Benchmark objects, each containing results
                       for a different vectorizer.
    :param labels: A list of string labels, corresponding one-to-one with the
                   `benchmarks` list, used for plot legends.
    :param output_dir: Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(benchmarks) != len(labels):
        raise ValueError("The 'benchmarks' list and 'labels' list must have the same number of elements.")

    sns.set_theme(style="whitegrid", palette="deep")

    model_specific_colors = {
        "Qwen38B": "#B19CD9",
        "OpenAI": "#E980A2",
        "Gemini": "#5EB5A6",
        "AllMiniLM": "#F29C7B",
        "LinqEmbedMistral": "#8A2BE2"
    }

    plot_palette = sns.color_palette([model_specific_colors.get(label, "#CCCCCC") for label in labels])
    sns.set_palette(plot_palette)

    all_cohort_labels = set()
    for benchmark in benchmarks:
        if benchmark.results_geras:
            all_cohort_labels.add(benchmark.results_geras.cohort_label)
        if benchmark.results_prevent_dementia:
            all_cohort_labels.add(benchmark.results_prevent_dementia.cohort_label)
        if benchmark.results_prevent_ad:
            all_cohort_labels.add(benchmark.results_prevent_ad.cohort_label)
        if benchmark.results_emif:
            all_cohort_labels.add(benchmark.results_emif.cohort_label)

    sorted_cohort_labels = sorted(list(all_cohort_labels))

    for cohort_label in sorted_cohort_labels:
        # --- Collect data for sorting first ---
        plot_data = []

        for i, (benchmark, label_name) in enumerate(zip(benchmarks, labels)):
            current_cohort_result: BenchmarkResult = None
            if cohort_label == "GERAS" and benchmark.results_geras and benchmark.results_geras.cohort_label == cohort_label:
                current_cohort_result = benchmark.results_geras
            elif cohort_label == "PREVENT Dementia" and benchmark.results_prevent_dementia and benchmark.results_prevent_dementia.cohort_label == cohort_label:
                current_cohort_result = benchmark.results_prevent_dementia
            elif cohort_label == "PREVENT-AD" and benchmark.results_prevent_ad and benchmark.results_prevent_ad.cohort_label == cohort_label:
                current_cohort_result = benchmark.results_prevent_ad
            elif cohort_label == "EMIF" and benchmark.results_emif and benchmark.results_emif.cohort_label == cohort_label:
                current_cohort_result = benchmark.results_emif

            if current_cohort_result and current_cohort_result.precisions and current_cohort_result.recalls:
                recalls = current_cohort_result.recalls
                precisions = current_cohort_result.precisions

                if len(recalls) > 1 and recalls[0] > recalls[-1]:
                    recalls = recalls[::-1]
                    precisions = precisions[::-1]

                auprc = current_cohort_result.auc
                color = plot_palette[i]

                plot_data.append((auprc, label_name, recalls, precisions, color))
            else:
                # Changed to a more informative logging if you want to keep track
                # Consider using Python's `logging` module for more robust logging
                print(
                    f"INFO: No PR data for {label_name} on cohort {cohort_label}. Skipping this model for this cohort's plot."
                )

        # --- Only create and save the plot if there is data to plot ---
        if not plot_data:
            print(
                f"WARNING: No PR data found for any model on cohort {cohort_label}. Skipping plot generation for this cohort.")
            continue  # Skip to the next cohort

        # If we reached here, plot_data is not empty, so proceed to plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve: {cohort_label}', fontsize=14)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # Sort plot_data by AUPRC in descending order
        plot_data.sort(key=lambda x: x[0], reverse=True)

        # Plot in sorted order
        for auprc, label_name, recalls, precisions, color in plot_data:
            ax.plot(recalls, precisions, drawstyle='steps-post',
                    label=f'{label_name} (AUPRC = {auprc:.4f})',
                    color=color,
                    linewidth=2)

        ax.legend(loc='upper right', frameon=True, fontsize=10)

        filename = f"pr_curve_comparison_{cohort_label.replace(' ', '_').lower()}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved PR curve comparison for {cohort_label} to {path}")


# --- Your Existing Loading Code and Call ---

# Load your benchmark results from pickle files
try:
    results_openai = pickle.load(open("results/OpenAI.pkl", "rb"))
    results_gemini = pickle.load(open("results/Gemini.pkl", "rb"))
    results_allminilm = pickle.load(open("results/AllMiniLM.pkl", "rb"))
    results_qwen = pickle.load(open("results/Qwen38B.pkl", "rb"))
    results_linq = pickle.load(open("results/LinqEmbedMistral.pkl", "rb"))
except FileNotFoundError as e:
    print(
        f"Error loading pickle file: {e}. Please ensure 'results/' directory exists and contains all required .pkl files.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading pickle files: {e}")
    exit()

# Create the list of Benchmark objects
benchmarks = [results_qwen, results_openai, results_gemini, results_allminilm, results_linq]

# Create the corresponding list of labels (ensure order matches 'benchmarks' for initial color assignment)
labels = ["Qwen38B", "OpenAI", "Gemini", "AllMiniLM", "LinqEmbedMistral"]

# Call the plotting function
plot_prauc_curves_per_cohort(benchmarks, labels, output_dir="plots/benchmark_comparison")