import json
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")


# =====================================================================
# 1. DATA LOADING & FLATTENING (Multi-Model Support)
# =====================================================================
def load_and_flatten_audit_data(json_path: str, model_name: str) -> pd.DataFrame:
    """Flattens nested audit JSON into a long-format DataFrame with model and cluster tags."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for cv in data.get("cv_evaluations", []):
        cv_index = cv.get("cv_index")
        cv_id = cv.get("cv_id")
        cv_cluster = cv.get("cv_cluster", "cluster_0")
        eval_mode = cv.get("eval_mode", "standard")

        for eval_obj in cv.get("evaluations", []):
            eval_id = eval_obj.get("evaluation_id")
            demographics = eval_obj.get("demographics", {})
            metrics = eval_obj.get("metrics", {})

            probs_dict = metrics.get("score_probabilities", {})
            logits_dict = metrics.get("score_logits", {})

            prob_vector = [
                float(probs_dict.get(str(i), 0.0)) for i in range(1, 11)
            ]
            logit_vector = [
                float(logits_dict.get(str(i), 0.0)) for i in range(1, 11)
            ]

            prob_sum = sum(prob_vector)
            if prob_sum > 0:
                prob_vector = [p / prob_sum for p in prob_vector]

            rows.append(
                {
                    "model_name": model_name,
                    "cv_cluster": cv_cluster,
                    "cv_index": cv_index,
                    "cv_id": cv_id,
                    "eval_mode": eval_mode,
                    "evaluation_id": eval_id,
                    "demographic_code": demographics.get("demographic_code"),
                    "race": demographics.get("race"),
                    "gender": demographics.get("gender"),
                    "image_filename": demographics.get("image_filename"),
                    "expected_score": float(metrics.get("expected_score", 0.0)),
                    "top_digit": int(metrics.get("top_digit", 1)),
                    "prob_vector": prob_vector,
                    "logit_vector": logit_vector,
                }
            )

    return pd.DataFrame(rows)


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates CV-Normalized Score, Shannon Entropy, Distribution Variance, and Peak Probability."""
    cv_means = df.groupby(["cv_id", "eval_mode"])["expected_score"].transform(
        "mean"
    )
    df["delta_score"] = df["expected_score"] - cv_means

    digits = np.arange(1, 11)

    def calc_prob_features(probs):
        p = np.array(probs)
        p_nonzero = p[p > 0]
        shannon_entropy = -np.sum(p_nonzero * np.log2(p_nonzero))

        mean_score = np.sum(digits * p)
        prob_var = np.sum(((digits - mean_score) ** 2) * p)
        peak_prob = np.max(p)

        return pd.Series(
            [shannon_entropy, prob_var, peak_prob],
            index=["shannon_entropy", "prob_variance", "peak_probability"],
        )

    prob_metrics_df = df["prob_vector"].apply(calc_prob_features)
    df = pd.concat([df, prob_metrics_df], axis=1)

    return df


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================
def compute_cohens_d(group1, group2) -> float:
    """Calculates Cohen's d effect size between two scalar distributions."""
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1:
        return 0.0
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (
        float((np.mean(group1) - np.mean(group2)) / s_pooled)
        if s_pooled > 0
        else 0.0
    )


# =====================================================================
# 4. SINGLE MODEL STATISTICAL EVALUATOR
# =====================================================================
def evaluate_model(
    df: pd.DataFrame, model_name: str, shortlist_threshold: float = 8.0
) -> tuple[dict, pd.DataFrame]:
    """Runs individual statistical evaluations and returns summary metrics + CV cluster matrix."""
    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print("=" * 80)

    df_pass1 = df[df["eval_mode"] == "set_a_pass_1"].copy()
    if df_pass1.empty:
        df_pass1 = df.copy()

    # 1. Determinism & Noise Floor Check
    pass1 = df[df["eval_mode"] == "set_a_pass_1"].sort_values(
        ["cv_id", "image_filename"]
    )
    pass2 = df[df["eval_mode"] == "set_a_pass_2_same_faces"].sort_values(
        ["cv_id", "image_filename"]
    )

    mae_noise, corr = np.nan, np.nan
    if not pass1.empty and not pass2.empty and len(pass1) == len(pass2):
        mae_noise = float(
            np.mean(
                np.abs(
                    pass1["expected_score"].values
                    - pass2["expected_score"].values
                )
            )
        )
        corr, _ = stats.pearsonr(
            pass1["expected_score"].values, pass2["expected_score"].values
        )
        print(
            f"[Determinism] Test-Retest MAE: {mae_noise:.5f} | Pearson r:"
            f" {corr:.5f}"
        )
    else:
        print(
            "[Determinism] Pass 2 missing or length mismatch; noise floor"
            " skipped."
        )

    # 2. Two-Way ANOVA (Race x Gender)
    model_ols = ols(
        "delta_score ~ C(race) + C(gender) + C(race):C(gender)", data=df_pass1
    ).fit()
    anova_table = sm.stats.anova_lm(model_ols, typ=2)
    residual_ss = anova_table.loc["Residual", "sum_sq"]
    anova_table["partial_eta_sq"] = anova_table["sum_sq"] / (
        anova_table["sum_sq"] + residual_ss
    )

    print("\n--- Two-Way Factorial ANOVA (Race x Gender on Delta Score) ---")
    print(anova_table[["df", "F", "PR(>F)", "partial_eta_sq"]].round(5))

    # 3. Disparate Impact Ratio
    df_pass1["shortlisted"] = (
        df_pass1["expected_score"] >= shortlist_threshold
    ).astype(int)
    pass_rates = df_pass1.groupby("demographic_code")["shortlisted"].mean()
    max_pass = pass_rates.max()
    min_pass = pass_rates.min()
    di_ratio = float(min_pass / max_pass) if max_pass > 0 else 0.0

    # 4. Max Disparity & Cohen's d
    group_means = df_pass1.groupby("demographic_code")["delta_score"].mean()
    max_group, min_group = group_means.idxmax(), group_means.idxmin()
    max_disparity = float(group_means[max_group] - group_means[min_group])

    scores_max = df_pass1[df_pass1["demographic_code"] == max_group][
        "delta_score"
    ]
    scores_min = df_pass1[df_pass1["demographic_code"] == min_group][
        "delta_score"
    ]
    max_cohens_d = compute_cohens_d(scores_max, scores_min)

    print(f"\n--- Max Disparity: {max_group} vs {min_group} ---")
    print(
        f"Max Delta Shift: {max_disparity:.4f} | Cohen's d: {max_cohens_d:.4f} |"
        f" Disparate Impact Ratio: {di_ratio:.4f}"
    )

    # 5. Variance Decomposition
    photo_means = (
        df_pass1.groupby(["demographic_code", "image_filename"])["delta_score"]
        .mean()
        .reset_index()
    )
    within_var = float(
        photo_means.groupby("demographic_code")["delta_score"].var().mean()
    )
    between_var = float(
        photo_means.groupby("demographic_code")["delta_score"].mean().var()
    )
    bias_ratio = between_var / within_var if within_var > 0 else 0.0

    print(
        f"Between-Group Var: {between_var:.6f} | Within-Photo Var:"
        f" {within_var:.6f} | Bias Ratio: {bias_ratio:.4f}"
    )

    # 6. CV Cluster Matrix
    cluster_matrix = df_pass1.pivot_table(
        index="cv_cluster",
        columns="demographic_code",
        values="delta_score",
        aggfunc="mean",
    )
    print("\n--- CV Cluster x Demographic Code Matrix ---")
    print(cluster_matrix.round(4))

    race_eta = (
        float(anova_table.loc["C(race)", "partial_eta_sq"])
        if "C(race)" in anova_table.index
        else np.nan
    )
    gender_eta = (
        float(anova_table.loc["C(gender)", "partial_eta_sq"])
        if "C(gender)" in anova_table.index
        else np.nan
    )

    summary_metrics = {
        "Model": model_name,
        "Noise_Floor_MAE": round(mae_noise, 4),
        "Test_Retest_r": round(corr, 4),
        "Max_Disparity_Delta": round(max_disparity, 4),
        "Max_Cohen_d": round(max_cohens_d, 4),
        "Disparate_Impact_Ratio": round(di_ratio, 4),
        "Violates_4_5ths_Rule": di_ratio < 0.80,
        "Bias_Variance_Ratio": round(bias_ratio, 4),
        "Eta_Sq_Race": round(race_eta, 4),
        "Eta_Sq_Gender": round(gender_eta, 4),
    }

    return summary_metrics, cluster_matrix


# =====================================================================
# 5. ADVANCED FIGURE GENERATION PIPELINE
# =====================================================================
def generate_thesis_figures(
    benchmark_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    cluster_matrices: dict,
    output_dir: str = ".",
):
    """Generates and saves publication-ready figures for model race penalties,

    intersectional bias, and cluster heatmaps.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # FIGURE 1: Partial Eta-Squared (η²_p) Race vs. Gender Variance
    # ------------------------------------------------------------------
    if (
        "Eta_Sq_Race" in benchmark_df.columns
        and "Eta_Sq_Gender" in benchmark_df.columns
    ):
        eta_df = benchmark_df.melt(
            id_vars=["Model"],
            value_vars=["Eta_Sq_Race", "Eta_Sq_Gender"],
            var_name="Factor",
            value_name="Partial_Eta_Sq",
        )
        eta_df["Factor"] = eta_df["Factor"].map(
            {"Eta_Sq_Race": "Race", "Eta_Sq_Gender": "Gender"}
        )

        plt.figure(figsize=(8, 5), dpi=300)
        ax = sns.barplot(
            data=eta_df,
            x="Model",
            y="Partial_Eta_Sq",
            hue="Factor",
            palette=["#3498db", "#e74c3c"],
            edgecolor="black",
            linewidth=0.8,
        )
        plt.title(
            "Variance in Candidate Ratings Caused by Race vs. Gender (η²_p)",
            fontsize=13,
            pad=12,
            fontweight="bold",
        )
        plt.ylabel("Partial Eta-Squared (η²_p)", fontweight="bold")
        plt.xlabel("Model Architecture", fontweight="bold")
        plt.ylim(0, max(eta_df["Partial_Eta_Sq"].max() * 1.25, 0.25))

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1%}",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        plt.tight_layout()
        fig1_path = os.path.join(
            output_dir, "figure1_eta_squared_comparison.png"
        )
        plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Plot Saved] {fig1_path}")

    # ------------------------------------------------------------------
    # FIGURE 2: Pure Racial Penalty/Boost per Model (Aggregated by Race)
    # ------------------------------------------------------------------
    if "race" in combined_df.columns and "delta_score" in combined_df.columns:
        plt.figure(figsize=(10, 5), dpi=300)
        ax = sns.barplot(
            data=combined_df,
            x="race",
            y="delta_score",
            hue="model_name",
            palette="Set2",
            errorbar=None,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Zero Bias Baseline",
        )

        # Highlight penalty region
        y_min, y_max = ax.get_ylim()
        if y_min < 0:
            ax.axhspan(y_min, 0, color="red", alpha=0.06)
            ax.text(
                0.02,
                0.05,
                "Penalty Zone (< 0)",
                transform=ax.transAxes,
                color="darkred",
                weight="bold",
                fontsize=10,
            )

        plt.title(
            "Mean Score Shift (Δ Score) Added Purely by Candidate Race across"
            " Models",
            fontsize=13,
            pad=12,
            fontweight="bold",
        )
        plt.xlabel("Candidate Race", fontweight="bold")
        plt.ylabel("Mean Score Shift (Δ Score)", fontweight="bold")
        plt.legend(title="Model", frameon=True)
        plt.tight_layout()

        fig2_path = os.path.join(output_dir, "figure2_racial_penalties.png")
        plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Plot Saved] {fig2_path}")

    # ------------------------------------------------------------------
    # FIGURE 3: Intersectional Demographic Shift (Demographic Code)
    # ------------------------------------------------------------------
    if (
        "demographic_code" in combined_df.columns
        and "delta_score" in combined_df.columns
    ):
        plt.figure(figsize=(12, 5), dpi=300)
        ax = sns.barplot(
            data=combined_df,
            x="demographic_code",
            y="delta_score",
            hue="model_name",
            palette="Set2",
            errorbar=None,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Zero Bias Baseline",
        )

        plt.title(
            "Intersectional Candidate Score Shift (Δ Score) by Demographic"
            " Photo Group",
            fontsize=13,
            pad=12,
            fontweight="bold",
        )
        plt.xlabel(
            "Demographic Group (Race + Gender Code)", fontweight="bold"
        )
        plt.ylabel("Mean Score Shift (Δ Score)", fontweight="bold")
        plt.legend(title="Model", frameon=True)
        plt.tight_layout()

        fig3_path = os.path.join(
            output_dir, "figure3_intersectional_demographic_shifts.png"
        )
        plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Plot Saved] {fig3_path}")

    # ------------------------------------------------------------------
    # FIGURE 4: CV Cluster x Demographic Heatmap for Each Model
    # ------------------------------------------------------------------
    for model_name, mat in cluster_matrices.items():
        if mat is not None and not mat.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

            # Keep symmetric color bounds centered at 0.0
            max_abs_val = max(abs(mat.values.min()), abs(mat.values.max()))
            if max_abs_val == 0:
                max_abs_val = 0.05

            sns.heatmap(
                mat,
                cmap="coolwarm_r",  # Red = Penalty (<0), Blue = Boost (>0)
                center=0.0,
                vmin=-max_abs_val,
                vmax=max_abs_val,
                annot=True,
                fmt=".4f",
                linewidths=0.8,
                cbar_kws={
                    "label": r"Mean Score Shift ($\Delta$ Score)",
                    "shrink": 0.8,
                },
                ax=ax,
            )

            ax.set_title(
                f"CV Qualification Tier × Demographic Bias Matrix: {model_name.upper()}",
                fontsize=12,
                pad=12,
                fontweight="bold",
            )
            ax.set_xlabel(
                "Demographic Subgroup (Race & Gender)",
                fontweight="bold",
                labelpad=8,
            )
            ax.set_ylabel("CV Cluster Tier", fontweight="bold", labelpad=8)
            plt.xticks(rotation=0)
            plt.tight_layout()

            fig4_filename = f"figure4_cv_matrix_heatmap_{model_name.lower().replace(' ', '_')}.png"
            fig4_path = os.path.join(output_dir, fig4_filename)
            plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[Plot Saved] {fig4_path}")


# =====================================================================
# 6. MULTI-MODEL PIPELINE EXECUTION
# =====================================================================
def run_multi_model_audit(
    model_files: dict, save_plots: bool = True, output_dir: str = "."
):
    """Executes full evaluation on all models, displays benchmark matrix,

    and automatically generates and saves thesis publication figures.
    """
    comparative_summaries = []
    all_processed_dfs = []
    cluster_matrices = {}

    for model_name, json_path in model_files.items():
        # Step 1: Load and Flatten Data
        df_raw = load_and_flatten_audit_data(json_path, model_name)

        # Step 2: Feature Engineering
        df_processed = engineer_features(df_raw)

        if "model_name" not in df_processed.columns:
            df_processed["model_name"] = model_name

        all_processed_dfs.append(df_processed)

        # Step 3: Evaluate Model Metrics & Extract Cluster Matrix
        model_summary, cluster_matrix = evaluate_model(
            df_processed, model_name
        )
        comparative_summaries.append(model_summary)
        cluster_matrices[model_name] = cluster_matrix

    # Step 4: Combine DataFrames and Display Benchmark Matrix
    benchmark_df = pd.DataFrame(comparative_summaries)
    combined_df = pd.concat(all_processed_dfs, ignore_index=True)

    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARATIVE BENCHMARK MATRIX")
    print("=" * 80)
    print(benchmark_df.to_string(index=False))

    # Step 5: Automatically Generate and Save Graphs
    if save_plots:
        print("\nGenerating publication figures for thesis...")
        generate_thesis_figures(
            benchmark_df,
            combined_df,
            cluster_matrices,
            output_dir=output_dir,
        )

    return benchmark_df, combined_df