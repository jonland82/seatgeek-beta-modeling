# Scaled Beta Models and Feature Dilution for Dynamic Ticket Pricing
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/Data-CC--BY--NC%204.0-lightgrey.svg)](DATA_LICENSE)

**Jonathan R. Landers, 2025**
---

### [Manuscript PDF](./seatgeek-beta-modeling.pdf)

## Introduction

This research presents a novel framework for analyzing dynamic ticket pricing in the secondary resale market, leveraging a time-series dataset curated from the SeatGeek API. By modeling ticket price distributions as scaled Beta distributions, we estimate shape parameters (α, β) using a hybrid of quantile matching and the method of moments, enabling accurate recovery of distributional patterns from limited summary statistics (minimum, maximum, mean, median). The approach captures unique economic signatures of performing artists, facilitating high-accuracy artist classification with Random Forest models. Additionally, we demonstrate that zero-variance (constant-value) features act as implicit regularizers in Random Forest ensembles, diluting the dominance of prominent predictors, promoting tree variety, and enhancing model robustness. Theoretical results link classification accuracy to distributional fidelity via Jensen-Shannon divergence, while practical case studies (e.g., Ed Sheeran vs. Beyoncé, Dropkick Murphys vs. The Avett Brothers) highlight real-world applications in market forecasting and pricing optimization. This framework not only advances time-series classification but also offers broader implications for financial modeling and demand forecasting.

---

## Summary

This repository supports the paper's two-part framework for modeling dynamic ticket pricing and enhancing Random Forest classification, leveraging a year-long dataset retrieved from the SeatGeek API. The framework addresses challenges in handling compressed time-series data and improving classifier robustness:

1. **Distribution Recovery from Limited Statistics**:  
   Each daily ticket price snapshot is modeled as a scaled Beta distribution. Using only four summary statistics (minimum, maximum, mean, and median), closed-form formulas estimate the shape parameters α and β via a composite quantile and moment-matching approach. This yields a compact, six-dimensional feature vector per event (min, max, mean, median, α, β), capturing the underlying pricing distribution's shape and scale.

2. **Robust Classification with Implicit Regularization**:  
   Incorporating the estimated α and β parameters into Random Forest artist classifiers significantly improves pairwise classification accuracy across model instances. Additionally, augmenting the feature set with zero-variance (constant-value) columns acts as an implicit regularizer, diluting the dominance of high-ranking predictors (e.g., mean, median). This leads to deeper, more varied trees, reduced inter-tree correlation, and enhanced generalization, as demonstrated on both the ticket pricing dataset and the UCI handwritten digits benchmark.

A theoretical link connects these components: Proposition 1 with Theorems 2 and 3 establish that classification accuracy improvements imply quadratic convergence  of the estimated Beta distribution to the true distribution under Jensen-Shannon divergence. Practical case studies (Ed Sheeran vs. Beyoncé; Dropkick Murphys vs. The Avett Brothers) highlight how α, β, and zero-variance regularization resolve subtle misclassifications, revealing artist-specific pricing signatures.

---

## Main Contributions

- **Closed-Form Beta Parameter Estimation**:  
  Developed a method to estimate scaled Beta distribution parameters (α, β) using only four summary statistics (min, max, mean, median). This approach, detailed in Section 4, leverages composite quantile and moment matching to recover distributional shapes, enabling robust analysis when raw price data is unavailable.

- **Curated SeatGeek Dataset**:  
  Compiled a comprehensive dataset of approximately 130,000 events, 15,400 artists, and 6,700 venues, with daily pricing snapshots from May 2023 to 2024. A statistically transformed subset of the event data that spans 954 artists was analyzed in this study and is made available for reproducibility at **[`event_labels_1_18_2025_last_N_days.csv`](./event_labels_1_18_2025_last_N_days.csv)**. Utilities for data creation and analysis are provided as outlined below.

- **Enhanced Random Forest Accuracy**:  
  Demonstrated that augmenting feature vectors with α and β parameters improves artist classification accuracy, with a statistically significant win rate, as shown in Table 2. This is attributed to the enriched representation of pricing dynamics.

- **Zero-Variance Feature Regularization**:  
  Introduced a novel implicit regularization technique by adding zero-variance features, which dilutes the selection probability of dominant predictors (Theorem 4). This results in deeper trees, higher ensemble variety, and improved accuracy on both ticket and digit datasets (Table 3). In addition, theoretical arguments prove the expanded regularization search space (Theorem 7). 

- **Information-Theoretic Validation**:  
  Established a theoretical connection between classification accuracy and distributional fidelity (Section 4.4). We prove that the classification error rate  implies an estimation error bound that leads to quadratic convergence in Jensen-Shannon divergence, validating the accuracy of the estimated Beta parameters.

- **Real-World Interpretability**:  
  Case studies (Figures 4 and 11) demonstrate how the framework captures artist-specific pricing signatures, correcting misclassifications (e.g., Ed Sheeran vs. Beyoncé, Dropkick Murphys vs. The Avett Brothers). These insights align with real-world ticket pricing trends, offering applications in market forecasting, pricing optimization, and economic analysis of the secondary ticket market.

---

## Key Notebooks & Artifacts

This repository contains all code and data necessary to reproduce the experimental results presented in the paper. The following Jupyter notebooks are organized to align with specific sections, figures, and tables in the manuscript:

- **[`dataset_stats.ipynb`](./dataset_stats.ipynb)**: Computes general statistics for the raw SeatGeek dataset, as referenced in Section 3.1. It provides insights into the dataset's scale, covering approximately 130,000 events, 15,400 artists, and 6,700 venues across the United States from May 2023 to 2024.
- **[`create_seatgeek_training_dataset.ipynb`](./create_seatgeek_training_dataset.ipynb)**: Processes raw SeatGeek API data to generate the primary machine learning dataset ([`event_labels_1_18_2025_last_N_days.csv`](./event_labels_1_18_2025_last_N_days.csv)) and produces Figure 1a, illustrating ticket price trends over time for a specific event (e.g., Buddy Guy at Wilbur Theatre, Boston, MA, 10/3/2023).
- **[`visualize_scaled_beta_distribution.ipynb`](./visualize_scaled_beta_distribution.ipynb)**: Generates Figure 1b, visualizing the estimated scaled Beta distribution parameters (α, β, mean, median, min, max) for the same Buddy Guy event, highlighting the economic signature of the pricing distribution.
- **[`statistical_significance.ipynb`](./statistical_significance.ipynb)**: Implements statistical significance tests for Random Forest model comparisons, producing results for Tables 2 and 3, which report metrics such as Z-scores and p-values for the ticket and digit datasets.
- **[`random_forest_on_tickets.ipynb`](./random_forest_on_tickets.ipynb)**: Trains and evaluates Random Forest models on the ticket pricing dataset, generating Figures 3a, 3b, 5a, 6a, 7a, 8a, and 9a. These figures illustrate model performance, feature usage, tree depth, and accuracy trends for artist classification.
- **[`random_forest_on_digits.ipynb`](./random_forest_on_digits.ipynb)**: Applies Random Forest models to the UCI handwritten digits dataset, producing Figures 5b, 6b, 7b, 8b, and 9b, demonstrating the generalizability of the zero-variance regularization approach.
- **[`kde_for_distribution_params.ipynb`](./kde_for_distribution_params.ipynb)**: Generates kernel density estimation (KDE) plots for distribution parameters, producing Figures 2 and 10. These figures compare feature distributions (e.g., α, β) across artists like Drake, Olivia Rodrigo, Dropkick Murphys, and The Avett Brothers, using Hellinger Distance and Jensen-Shannon divergence metrics.

### Case Studies

- **[`ed_sheeran_vs_beyonce.ipynb`](./ed_sheeran_vs_beyonce.ipynb)**: Analyzes the misclassification of an Ed Sheeran concert as a Beyoncé event, generating Figures 4a and 4b. These figures contrast basic statistical features with distribution-augmented features (α, β), demonstrating how the latter corrects misclassifications by capturing nuanced pricing patterns.
- **[`dm_vs_ab_regularization_improvements_analysis.ipynb`](./dm_vs_ab_regularization_improvements_analysis.ipynb)**: Investigates the classification of a Dropkick Murphys concert mislabeled as The Avett Brothers, producing Figures 11a and 11b. These figures illustrate how zero-variance feature regularization enhances feature selection (e.g., increasing β's prominence) to correct misclassifications. Figure 10 is also related to this example, but created by [`kde_for_distribution_params.ipynb`](./kde_for_distribution_params.ipynb).

### Data

- **[`event_labels_1_18_2025_last_N_days.csv`](./event_labels_1_18_2025_last_N_days.csv)**: The primary machine learning dataset, derived from raw SeatGeek API data using [`create_seatgeek_training_dataset.ipynb`](./create_seatgeek_training_dataset.ipynb). 
- **Intermediate Results**: Cached datasets to streamline reproducibility and avoid re-running computationally expensive tasks:
  - **[`total_correct.pkl`](./total_correct.pkl)**
  - **[`accuracy_no_const_2.json`](./accuracy_no_const_2.json)**
  - **[`accuracy_const_2.json`](./accuracy_const_2.json)**
  - **[`results_df_4_28_2024_regularization.csv`](./results_df_4_28_2024_regularization.csv)**
  - **[`results_df_2_6_2024.csv`](./results_df_2_6_2024.csv)**

## Acknowledgements

The author used publicly available event data accessed via the SeatGeek API (SeatGeek, Inc.) in accordance with SeatGeek’s API Terms of Use. SeatGeek is not affiliated with this research and does not endorse it. All trademarks and content remain the property of their respective owners. Proper attribution is provided at [seatgeek.com](https://seatgeek.com) as required. Raw API data is not redistributed per licensing requirements. Only derived statistical features and methodology code are available in this repository.

The author also used standard computational tools and programming libraries (including Python packages and the large language model ChatGPT-4 by OpenAI) for coding support, symbolic manipulation, and editorial refinement during the preparation of this manuscript. All outputs were reviewed by the author for accuracy and appropriateness.
