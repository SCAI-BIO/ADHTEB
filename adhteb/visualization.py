from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

from pandas.plotting import table


def plot_top_n_accuracies(
        accuracies: Dict[str, List[float]],
        save_path: Optional[str] = None,
        palette: Optional[str] = "Set2",
        vectorizer_name: Optional[str] = None,
) -> None:
    """
    Plot cumulative top-N accuracies for a single vectorizer with a modern style.

    :param accuracies: Output from Benchmark.get_accuracies(n), i.e., Dict[cohort → List[accuracy@1...n]]
    :param title: Plot title.
    :param save_path: If provided, save the figure to this path instead of showing it.
    :param palette: Optional seaborn color palette name.
    """
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    n = len(next(iter(accuracies.values())))
    x = list(range(1, n + 1))

    color_palette = sns.color_palette(palette, n_colors=len(accuracies))

    for (cohort_name, acc_list), color in zip(accuracies.items(), color_palette):
        plt.plot(x, acc_list, marker='o', linewidth=2.5, label=cohort_name, color=color)

    title: str = "Top-N Accuracy per Cohort " + vectorizer_name if vectorizer_name else "Top-N Accuracy per Cohort"

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Top-N", fontsize=13)
    plt.ylabel("Cumulative Accuracy", fontsize=13)
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Cohort", fontsize=11, title_fontsize=12, loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + "/" + "overall_" + vectorizer_name + ".png", dpi=300)
        plt.close()
    else:
        plt.show()


def summarize_top1_accuracies(all_accuracies: Dict[str, Dict[str, List[float]]], save_path=None) -> pd.DataFrame:
    """
    Summarize top-1 accuracies for multiple vectorizers in a table format.

    :param all_accuracies: Dict[vectorizer_name → Dict[cohort → List[float (length N)]]]
    :param save_path: If provided, save the summary table as an image in the specified directory.
    :return: DataFrame with cohorts as rows, vectorizers as columns, and top-1 accuracies as values.
    """
    # Create a DataFrame with cohorts as rows and vectorizers as columns
    summary = {
        vectorizer_name: {cohort: round(acc_list[0], 2) for cohort, acc_list in cohort_dict.items()}
        for vectorizer_name, cohort_dict in all_accuracies.items()
    }
    df_summary = pd.DataFrame(summary).sort_index()

    # Add bold formatting for the largest entry per row
    def format_bold_row(row):
        max_value = row.max()
        return [f"**{value}**" if value == max_value else f"{value}" for value in row]

    formatted_data = df_summary.apply(format_bold_row, axis=1).to_list()
    df_summary_formatted = pd.DataFrame(formatted_data, index=df_summary.index, columns=df_summary.columns)

    if save_path:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.axis('off')  # Turn off the axis
        tbl = table(ax, df_summary_formatted, loc='center', cellLoc='center')  # Render table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.2)  # Scale table for better readability
        plt.savefig(f"{save_path}/summary_table.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    return df_summary


def plot_topn_per_cohort(
        all_accuracies: Dict[str, Dict[str, List[float]]],
        title_prefix: str = "Top-N Accuracy",
        palette: Optional[str] = "Set2",
        save_path: Optional[str] = None
) -> None:
    """
    Plot Top-N accuracy curves per cohort comparing multiple vectorizers.

    :param all_accuracies: Dict[vectorizer_name → Dict[cohort → List[float]]]
    :param title_prefix: Prefix for the plot titles.
    :param palette: Seaborn color palette.
    """
    sns.set(style="whitegrid", font_scale=1.1)

    # Get all cohorts from any one vectorizer
    example_vectorizer = next(iter(all_accuracies.values()))
    cohorts = list(example_vectorizer.keys())
    n = len(next(iter(example_vectorizer.values())))  # Number of top-N levels
    x = list(range(1, n + 1))

    for cohort in cohorts:
        plt.figure(figsize=(10, 6))
        color_palette = sns.color_palette(palette, n_colors=len(all_accuracies))

        for (vectorizer_name, vectorizer_accuracies), color in zip(all_accuracies.items(), color_palette):
            y = vectorizer_accuracies[cohort]
            plt.plot(x, y, marker='o', label=vectorizer_name, linewidth=2.5, color=color)

        plt.title(f"{title_prefix}: {cohort}", fontsize=15, fontweight='bold')
        plt.xlabel("Top-N", fontsize=12)
        plt.ylabel("Cumulative Accuracy", fontsize=12)
        plt.xticks(x)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Vectorizer", fontsize=10, title_fontsize=11, loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + "/" + cohort + ".png", dpi=300)
            plt.close()
        else:
            plt.show()


def plot_precision_recall_curves(
        pr_results: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None,
        palette: Optional[str] = "Set2",
) -> None:
    """
    Plot Precision-Recall curves for each cohort and display the AUC in the legend.

    :param pr_results: Output from Benchmark.get_precision_recall(), i.e., Dict[cohort → {'precision': [...], 'recall': [...], 'thresholds': [...]}]
    :param save_path: If provided, save the figure to this path instead of showing it.
    :param palette: Optional seaborn color palette name.
    """
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    color_palette = sns.color_palette(palette, n_colors=len(pr_results))

    for (cohort_name, metrics), color in zip(pr_results.items(), color_palette):
        precision = metrics.get("precision", [])
        recall = metrics.get("recall", [])
        if not precision or not recall:
            continue
        # Compute AUC using trapezoidal rule: area under Precision-Recall curve
        auc = np.trapz(precision, recall)
        plt.plot(
            recall,
            precision,
            marker='o',
            linewidth=2.5,
            label=f"{cohort_name} (AUC={auc:.2f})",
            color=color
        )

    plt.title("Precision-Recall Curve per Cohort", fontsize=16, fontweight='bold')
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Cohort", fontsize=11, title_fontsize=12, loc="lower left")
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/precision_recall_curves.png", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_auprc_per_cohort(
        all_pr_results: Dict[str, Dict[str, Dict[str, List[float]]]],
        save_path: Optional[str] = None,
        palette: Optional[str] = "Set2",
) -> None:
    """Plot AUPRC (Precision-Recall curves) per cohort, comparing multiple vectorizers.

    :param all_pr_results: A nested dictionary containing precision and recall values as lists.
    :param save_path: Optional path to save the plots, defaults to None
    :param palette: Seaborn color palette for styling, defaults to "Set2"
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Get list of cohorts from the first vectorizer
    first_vectorizer = next(iter(all_pr_results.values()))
    cohorts = list(first_vectorizer.keys())

    for cohort in cohorts:
        plt.figure(figsize=(10, 6))
        color_palette = sns.color_palette(palette, n_colors=len(all_pr_results))

        for (vectorizer_name, vectorizer_results), color in zip(all_pr_results.items(), color_palette):
            cohort_data = vectorizer_results.get(cohort, {})
            precision = cohort_data.get("precision", [])
            recall = cohort_data.get("recall", [])
            if not precision or not recall:
                continue
            auc = np.trapz(precision, recall)
            plt.plot(recall, precision, label=f"{vectorizer_name} (AUC={auc:.2f})", linewidth=2.5, color=color)

        plt.title(f"{cohort}", fontsize=15, fontweight="bold")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Vectorizer", fontsize=10, title_fontsize=11, loc="upper right")
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/auprc_{cohort}.png", dpi=300)
            plt.close()
        else:
            plt.show()


# load oai from pickle
results_openai = pickle.load(open("results/OpenAI.pkl", "rb"))

print("")