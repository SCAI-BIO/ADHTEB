import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional


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
    'GERAS-I': [0.6363636363636364, 0.6363636363636364, 0.7272727272727273, 0.7272727272727273, 0.7272727272727273,
                0.7272727272727273, 0.7272727272727273, 0.7272727272727273, 0.7272727272727273, 0.7272727272727273,
                0.7272727272727273, 0.7272727272727273, 0.7272727272727273, 0.7272727272727273, 0.7272727272727273,
                0.7272727272727273, 0.8181818181818182, 0.9090909090909091, 0.9090909090909091, 0.9090909090909091],
    'GERAS-US': [0.7777777777777778, 0.7777777777777778, 0.7777777777777778, 0.7777777777777778, 0.7777777777777778,
                 0.7777777777777778, 0.7777777777777778, 0.7777777777777778, 0.7777777777777778, 0.8888888888888888,
                 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0],
    'GERAS-J': [0.6, 0.8, 0.85, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    'GERAS-II': [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.875, 0.875, 0.875, 0.875, 0.875, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0],
    'PREVENT Dementia': [0.5217391304347826, 0.6086956521739131, 0.6521739130434783, 0.6956521739130435,
                         0.782608695652174, 0.782608695652174, 0.782608695652174, 0.782608695652174, 0.8260869565217391,
                         0.8695652173913043, 0.9130434782608695, 0.9130434782608695, 0.9130434782608695,
                         0.9130434782608695, 0.9130434782608695, 0.9130434782608695, 0.9130434782608695,
                         0.9130434782608695, 0.9130434782608695, 0.9130434782608695]}

results_linq = {
    'GERAS-I': [0.4166666666666667, 0.5, 0.5833333333333334, 0.5833333333333334, 0.6666666666666666, 0.6666666666666666,
                0.6666666666666666, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                0.8333333333333334],
    'GERAS-US': [0.3, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    'GERAS-J': [0.38095238095238093, 0.5238095238095238, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
                0.6666666666666666, 0.7142857142857143, 0.7142857142857143, 0.7619047619047619, 0.7619047619047619,
                0.7619047619047619, 0.7619047619047619, 0.7619047619047619, 0.7619047619047619, 0.7619047619047619,
                0.7619047619047619, 0.7619047619047619, 0.7619047619047619, 0.8095238095238095, 0.8095238095238095],
    'GERAS-II': [0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 0.6666666666666666,
                 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
                 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
                 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
    'PREVENT Dementia': [0.1320754716981132, 0.1320754716981132, 0.18867924528301888, 0.20754716981132076,
                         0.20754716981132076, 0.20754716981132076, 0.20754716981132076, 0.20754716981132076,
                         0.22641509433962265, 0.22641509433962265, 0.22641509433962265, 0.22641509433962265,
                         0.22641509433962265, 0.24528301886792453, 0.24528301886792453, 0.24528301886792453,
                         0.2830188679245283, 0.2830188679245283, 0.3018867924528302, 0.3018867924528302]}

plot_top_n_accuracies(accuracies=results_linq, vectorizer_name="Linq-Embed-Mistral", save_path="plots")
plot_top_n_accuracies(accuracies=results_openai, vectorizer_name="OpenAI", save_path="plots")

all_accuracies = {
    "OpenAI": results_openai,
    "Linq-Embed-Mistral": results_linq,
}

df_summary = summarize_top1_accuracies(all_accuracies)
print(df_summary)

plot_topn_per_cohort(all_accuracies, title_prefix="Vectorizer Benchmark", save_path="plots")
