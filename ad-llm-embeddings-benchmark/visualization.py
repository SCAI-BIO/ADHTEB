import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional


def plot_top_n_accuracies(
        accuracies: Dict[str, List[float]],
        title: str = "Top-N Accuracy per Cohort",
        save_path: Optional[str] = None,
        palette: Optional[str] = "Set2"
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

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Top-N", fontsize=13)
    plt.ylabel("Cumulative Accuracy", fontsize=13)
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Cohort", fontsize=11, title_fontsize=12, loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + "/" + "overall_openai.png", dpi=300)
        plt.close()
    else:
        plt.show()


def summarize_top1_accuracies(all_accuracies: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Summarize top-1 accuracies for multiple vectorizers.

    :param all_accuracies: Dict[vectorizer_name → Dict[cohort → List[float (length N)]]]
    :return: DataFrame with columns: Cohort, Vectorizer, Top-1 Accuracy
    """
    summary = []
    for vectorizer_name, cohort_dict in all_accuracies.items():
        for cohort, acc_list in cohort_dict.items():
            summary.append({
                "Cohort": cohort,
                "Vectorizer": vectorizer_name,
                "Top-1 Accuracy": acc_list[0]  # top-1
            })
    return pd.DataFrame(summary).sort_values(by=["Cohort", "Vectorizer"])


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
results_openai = {
    'GERAS-I': [0.5833333333333334, 0.5833333333333334, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
                0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
    'GERAS-US': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8],
    'GERAS-J': [0.5714285714285714, 0.7619047619047619, 0.8095238095238095, 0.8095238095238095, 0.8571428571428571,
                0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571],
    'GERAS-II': [0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
                 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.7777777777777778],
    'PREVENT Dementia': [0.20754716981132076, 0.2830188679245283, 0.3018867924528302, 0.3018867924528302,
                         0.33962264150943394, 0.33962264150943394, 0.33962264150943394, 0.33962264150943394,
                         0.3584905660377358, 0.37735849056603776]}

title = "Top-N Cumulative Accuracy for OpenAI Vectorizer"

plot_top_n_accuracies(accuracies=results_openai, title=title, save_path="plots")

all_accuracies = {
    "OpenAI": results_openai,
    "MiniLM": results_openai,
    "BERT": results_openai
}

df_summary = summarize_top1_accuracies(all_accuracies)
print(df_summary)

plot_topn_per_cohort(all_accuracies, title_prefix="Vectorizer Benchmark", save_path="plots")
