# Algorithmic Bias Audit of Vision-Language Models in Automated Resume Screening

This repository contains an end-to-end audit framework to measure, quantify, and analyze demographic, racial, and intersectional biases in Vision-Language Models (VLMs) when scoring candidate resumes/CVs.

Instead of evaluating models on textual CVs alone, this project pairs standardized candidate face portraits with controlled resume documents to evaluate whether multimodal LLMs—such as **Qwen2.5-VL**, **Qwen3.5**, and **Phi-3.5 Vision**—introduce bias into candidate evaluations based on visual demographic attributes.

---

## 1. Project Overview

Automated hiring systems increasingly leverage Vision-Language Models (VLMs) to parse and grade multi-page candidate application materials. However, when visual candidate portraits or profile images are attached to application files, models may exhibit systemic bias.

This audit framework evaluates VLM bias by:
- **Testing Multimodal Models**: Auditing multimodal models including **Qwen2.5-VL**, **Qwen3.5**, and Microsoft's **Phi-3.5 Vision**.
- **Isolating Visual Demographic Effect**: Controlled pairing of identical CV text with diverse facial demographic profiles to measure score variation.
- **Logit-Level Evaluation**: Extracting full probability distributions over rating tokens ($1$ to $10$) rather than relying solely on generated score strings, eliminating generation noise.
- **Factorial Statistical Analysis**: Applying Two-Way Factorial ANOVA, Disparate Impact Ratios (EEOC 4/5ths Rule), Variance Decomposition, and Cohen's $d$ effect sizes.

---

## 2. Dataset Composition

The audit dataset combines standardized candidate face portraits with a curated, representative set of professional resumes:

1. **Facial Demographic Dataset**:
   - High-quality portrait images balanced across **4 Racial/Ethnic Groups** (Asian, Black, Latino, White, Indian) and **2 Genders** (Male, Female), yielding 10 distinct demographic codes (`AM`, `AF`, `BM`, `BF`, `LM`, `LF`, `WM`, `WF`, `IM`, `IF`).

2. **Resume/CV Corpus**:
   - A diverse collection of professional CVs spanning technical, managerial, and administrative disciplines.

---

## 3. Dataset Curation & Processing Workflow

To guarantee high evaluation quality and structured representation across candidate skill levels, the dataset was built through the following sequence:

### Processing Steps:
1. **Image Preprocessing**: Face portraits are resized to uniform resolution to prevent resolution or aspect ratio artifacts from influencing model vision encoders.
2. **Length & Quality Filtering**: Raw resumes are filtered to retain only complete, high-information CVs containing **over 2,000 characters** (eliminating truncated or sparse documents).
3. **Text-Only Qualification Tiering**: Resumes are evaluated by a text-only LLM (**Qwen 2.5 Text**) without images to rate candidate strength into 3 qualification tiers:
   - **Weak Tier**: Low experience / minimal skills
   - **Normal Tier**: Mid-level qualifications / standard industry experience
   - **Strong Tier**: Senior-level expertise / high achievement
4. **Clustering & Selection (36 Benchmark CVs)**:
   - Feature embeddings are extracted for resumes within each qualification tier.
   - **K-Means clustering** is performed to select **12 representative CVs per tier** (Cluster 0, Cluster 1, Cluster 2).
   - This yields a balanced set of **36 benchmark CVs** across the qualification spectrum.
5. **Data Pairing Assembly**: Each of the 36 baseline CVs is systematically paired with every facial portrait to form the audit evaluation matrix.

---

## 4. Evaluation Volume Breakdown

When auditing across the **3 qualification clusters** (Weak, Normal, Strong) and **10 demographic profiles**, the evaluation volume per model is calculated as follows:

| Component | Count / Factor | Description |
| :--- | :--- | :--- |
| **CV Qualification Tiers** | 3 Clusters | Weak, Normal, Strong |
| **CVs per Tier** | 12 Resumes | Selected via K-Means embedding clustering |
| **Total Benchmark Resumes** | **36 Unique CVs** | $3 \text{ Clusters} \times 12 \text{ CVs}$ |
| **Demographic Face Profiles** | **10 Portraits** | Standardized demographic groups |
| **Unique Evaluation Pairs** | **360 Pairs** | $36 \text{ Resumes} \times 10 \text{ Face Profiles}$ per pass |
| **Test-Retest Passes** | 3 Passes | Duplicate evaluation runs to establish noise floor and correlation |
| **Total Evaluations per Model** | **1080 Runs** | $360 \text{ Pairs} \times 2 \text{ Passes}$ |
| **Grand Total Across All 3 Models** | **3240 Evaluations** | $1080 \text{ Runs} \times 3 \text{ Models}$ (Qwen2.5-VL, Qwen3.5, Phi-3.5) |

---

## 5. Audit Execution & Logit Probability Evaluation

Rather than relying on raw generated text strings (which can suffer from generation instability), models are evaluated directly on the soft probability distribution output across rating tokens 1 through 10.

- **Expected Score**: Calculated by taking the weighted average of all score probabilities (1 to 10) produced by the model logits.
- **CV Score Shift**: Measures pure demographic bias by subtracting the resume's baseline average score from each individual evaluation score. This isolates candidate portrait effects from the underlying quality of the resume text.

---

## 6. Conceptual Guide to Statistical Metrics

Results across CV clusters are aggregated per model and summarized into ten statistical metrics:

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

---

## 7. Visualizations & Outputs

The pipeline automatically outputs publication-ready 300 DPI figures:
1. **`figure1_eta_squared_comparison.png`**: ANOVA effect sizes comparing Race vs. Gender variance per model.
2. **`figure2_racial_penalties.png`**: Mean score penalties and boosts aggregated purely by race with a zero-bias baseline.
3. **`figure3_intersectional_demographic_shifts.png`**: Intersectional breakdown across all 10 demographic codes.
4. **`figure4_cv_matrix_heatmap_<model>.png`**: Diverging heatmaps mapping score shifts across CV qualification tiers and demographic codes.

---

## 8. How to Run

To execute the audit pipeline, follow this workflow:

1. **Install Dependencies**: Execute the requirements file to install all necessary Python libraries and scientific computing packages into your environment.
2. **Configure Execution Flags**: Open the configuration file and select the appropriate execution flags for the modules you wish to run:
   - **Data Processing**: Enables image resizing, resume character filtering (>2,000 chars), tier classification, and K-Means clustering.
   - **Model Evaluation**: Toggles logit evaluation runs for Qwen2.5-VL, Qwen3.5, and Phi-3.5 Vision models.
   - **Post-Processing & Analytics**: Enables metric calculation, ANOVA testing, and publication figure generation.
3. **Execute Pipeline**: Run the `main.py` file to start the pipeline. Results, benchmark metrics, and output plots will automatically save to the designated output folder.

