# FraudLens: Cost-Sensitive Fraud Detection & Decision Intelligence

FraudLens is an end-to-end, mathematically rigorous machine learning pipeline. Rather than stopping at standard metrics (like PR-AUC), this project implements a **Decision Intelligence** layer that calibrates model probabilities and selects decision thresholds based on actual business constraints: investigation capacity and the financial cost of missed fraud versus false alarms.

## Core ML Engineering Highlights
* **Leakage-Proof Feature Engineering:** Implemented a custom Scikit-Learn Transformer to engineer cyclical time features (Sine/Cosine), Log-Amounts, and Hourly Z-Scores strictly within training folds to prevent temporal data leakage.
* **Imbalance Strategy Showdown:** Proved mathematically via 5-Fold Stratified Cross-Validation that algorithmic class weighting outperformed SMOTE (Synthetic Oversampling) by 7% PR-AUC for this specific feature space.
* **Probability Calibration:** Applied Isotonic Regression to map abstract XGBoost scores to literal real-world probabilities, improving the Brier Score by 7.3%.
* **Cost-Matrix Optimization:** Calculated the exact financial threshold (1.0%) that minimized the bank's dollar loss, assuming $25 per false alarm and $1,000 per missed fraud.
* **Statistical Rigor:** Generated 95% Confidence Intervals for the final test set using a 1,000-iteration Bootstrap Resampler to prove model stability.

## Tech Stack
* **Machine Learning:** Python, Scikit-Learn, XGBoost, Imbalanced-Learn
* **Data Processing:** Pandas, NumPy
* **Frontend:** Streamlit

## Repository Structure
* `src/fraudlens/data.py` - Strict 70/15/15 stratified data splitting.
* `src/fraudlens/features.py` - Custom BaseEstimator for leak-free feature engineering.
* `src/fraudlens/train.py` - 5-Fold CV pipelines and XGBoost hyperparameter tuning.
* `src/fraudlens/evaluate.py` - Isotonic Regression probability calibration.
* `src/fraudlens/decision_policy.py` - Bootstrap CIs and Financial Cost Matrix.