# Markdown

# Scaled Beta Models and Feature Dilution for Dynamic Ticket Pricing

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/Data-CC--BY--NC%204.0-lightgrey.svg)](DATA_LICENSE)

## **Jonathan R. Landers, 2025**

### [Manuscript PDF](./seatgeek-beta-modeling-v2.pdf)

## Introduction

This repository accompanies a three-part contribution: **(1)** a closed-form, constant-time estimator that recovers scaled-Beta parameters $(\alpha,\beta)$ from `{min, max, mean, median}`; **(2)** an accuracy–fidelity link showing classification error bounds total variation and contracts **quadratically** in Jensen–Shannon divergence in the small-error regime; and **(3)** an implicit regularization mechanism for Random Forests via zero-variance (constant-value) features that rebalance split selection, deepen/variate trees, and yield modest but reliable accuracy gains. Experiments use a year-long SeatGeek pricing dataset; a digits benchmark confirms generality.

---

## Overview & Contributions

1. **Closed-Form Distribution Recovery from Limited Statistics**
   Each daily snapshot is modeled as a **scaled Beta** on $[\text{min},\text{max}]$. From `{min, max, mean, median}`, closed-form formulas recover $(\alpha,\beta)$ via composite quantile (median) + moment (mean) matching, yielding a compact six-feature vector per event $(\text{min}, \text{max}, \text{mean}, \text{median}, \alpha, \beta)$. Adding $(\alpha,\beta)$ to Random Forests improves pairwise artist classification on time-series snapshots. Case studies (e.g., Ed Sheeran vs. Beyoncé) illustrate how $(\alpha,\beta)$ resolve subtle misclassifications.

2. **Accuracy–Fidelity Link (TV & JS, Quadratic Regime)**
   Improvements in Random Forest classification accuracy bound **total variation** and imply **quadratic contraction in Jensen–Shannon divergence** in the small-error regime—so modest accuracy gains signify disproportionately stronger distributional agreement when ground-truth densities are unavailable.

3. **Implicit Regularization via Zero-Variance Features**
   Adding **zero-variance (constant-value) features** **dilutes** over-dominant predictors and increases ensemble variety and expected depth. An approximation and corollaries provide **near-continuous control** of split-selection probabilities and accuracy expansion in expectation. Case studies (e.g., Dropkick Murphys vs. The Avett Brothers) demonstrate how $(\alpha,\beta)$ and the regularizer overcome nuanced errors.

**Dataset & Reproducibility**: ~130,000 events, 15,400 artists, and 6,700 venues (May 2023–2024). A transformed subset spanning 954 artists is available at **[`event_labels_1_18_2025_last_N_days.csv`](./event_labels_1_18_2025_last_N_days.csv)**. Utilities for creation and analysis are included.

---

## Key Notebooks & Artifacts

This repository contains all code and data necessary to reproduce the experimental results presented in the paper. The following Jupyter notebooks are organized to align with specific sections, figures, and tables in the manuscript:

* **[`dataset_stats.ipynb`](./dataset_stats.ipynb)**: Computes general statistics for the raw SeatGeek dataset, as referenced in Section 3.1. It provides insights into the dataset's scale, covering approximately 130,000 events, 15,400 artists, and 6,700 venues across the United States from May 2023 to 2024.
* **[`create_seatgeek_training_dataset.ipynb`](./create_seatgeek_training_dataset.ipynb)**: Processes raw SeatGeek API data to generate the primary machine learning dataset ([`event_labels_1_18_2025_last_N_days.csv`](./event_labels_1_18_2025_last_N_days.csv)) and produces Figure 1a, illustrating ticket price trends over time for a specific event (e.g., Buddy Guy at Wilbur Theatre, Boston, MA, 10/3/2023).
* **[`visualize_scaled_beta_distribution.ipynb`](./visualize_scaled_beta_distribution.ipynb)**: Generates Figure 1b, visualizing the estimated scaled Beta distribution parameters (α, β, mean, median, min, max) for the same Buddy Guy event, highlighting the economic signature of the pricing distribution.
* **[`statistical_significance.ipynb`](./statistical_significance.ipynb)**: Implements statistical significance tests for Random Forest model comparisons, producing results for Tables 2 and 3, which report metrics such as Z-scores and p-values for the ticket and digit datasets.
* **[`random_forest_on_tickets.ipynb`](./random_forest_on_tickets.ipynb)**: Trains and evaluates Random Forest models on the ticket pricing dataset, generating Figures 3a, 3b, 5a, 6a, 7a, 8a, and 9a. These figures illustrate model performance, feature usage, tree depth, and accuracy trends for artist classification.
* **[`random_forest_on_digits.ipynb`](./random_forest_on_digits.ipynb)**: Applies Random Forest models to the UCI handwritten digits dataset, producing Figures 5b, 6b, 7b, 8b, and 9b, demonstrating the generalizability of the zero-variance regularization approach.
* **[`kde_for_distribution_params.ipynb`](./kde_for_distribution_params.ipynb)**: Generates kernel density estimation (KDE) plots for distribution parameters, producing Figures 2 and 10. These figures compare feature distributions (e.g., α, β) across artists like Drake, Olivia Rodrigo, Dropkick Murphys, and The Avett Brothers, using Hellinger Distance and Jensen-Shannon divergence metrics.
* **[`accuracy_fidelity_theory_experiment_refined.ipynb`](./accuracy_fidelity_theory_experiment_refined.ipynb)**: Implements the **synthetic ground-truth validation** of the accuracy–fidelity link (Section 4.6): generates scaled-Beta ground truth, reconstructs $(\alpha,\beta)$ from `{min, max, mean, median}`, and plots **JS vs. TV** with two-sided bounds; produces the divergence figures referenced in Section 4.

### Case Studies

* **[`ed_sheeran_vs_beyonce.ipynb`](./ed_sheeran_vs_beyonce.ipynb)**: Analyzes the misclassification of an Ed Sheeran concert as a Beyoncé event, generating Figures 4a and 4b. These figures contrast basic statistical features with distribution-augmented features (α, β), demonstrating how the latter corrects misclassifications by capturing nuanced pricing patterns.
* **[`dm_vs_ab_regularization_improvements_analysis.ipynb`](./dm_vs_ab_regularization_improvements_analysis.ipynb)**: Investigates the classification of a Dropkick Murphys concert mislabeled as The Avett Brothers, producing Figures 11a and 11b. These figures illustrate how zero-variance feature regularization enhances feature selection (e.g., increasing β's prominence) to correct misclassifications. Figure 10 is also related to this example, but created by [`kde_for_distribution_params.ipynb`](./kde_for_distribution_params.ipynb).

### Data

* **[`event_labels_1_18_2025_last_N_days.csv`](./event_labels_1_18_2025_last_N_days.csv)**: The primary machine learning dataset, derived from raw SeatGeek API data using [`create_seatgeek_training_dataset.ipynb`](./create_seatgeek_training_dataset.ipynb).
* **Intermediate Results**: Cached datasets to streamline reproducibility and avoid re-running computationally expensive tasks:

  * **[`total_correct.pkl`](./total_correct.pkl)**
  * **[`accuracy_no_const_2.json`](./accuracy_no_const_2.json)**
  * **[`accuracy_const_2.json`](./accuracy_const_2.json)**
  * **[`results_df_4_28_2024_regularization.csv`](./results_df_4_28_2024_regularization.csv)**
  * **[`results_df_2_6_2024.csv`](./results_df_2_6_2024.csv)**

## Acknowledgements

The author used publicly available event data accessed via the SeatGeek API (SeatGeek, Inc.) in accordance with SeatGeek's API Terms of Use. SeatGeek is not affiliated with this research and does not endorse it. All trademarks and content remain the property of their respective owners. Proper attribution is provided at [seatgeek.com](https://seatgeek.com) as required. Raw API data is not redistributed per licensing requirements. Only derived statistical features and methodology code are available in this repository.

The author used standard computational tools and programming libraries,
including Python packages and a large language model (OpenAI),
to assist with code snippets, algebraic manipulation, and editorial refinement.
All content and interpretations were conceptualized, reviewed and finalized by the author.