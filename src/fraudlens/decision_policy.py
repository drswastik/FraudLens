# src/fraudlens/decision_policy.py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score

def evaluate_financial_cost(y_true, y_probs, thresholds, cost_fp=25, cost_fn=1000):
    """Calculates financial loss across all thresholds to find the optimum."""
    results = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        results.append((t, total_cost, tp, fp, fn))
    
    df_results = pd.DataFrame(results, columns=['Threshold', 'Total_Cost', 'TP', 'FP', 'FN'])
    # Return the row with the lowest financial cost
    return df_results.sort_values('Total_Cost').iloc[0]

def bootstrap_evaluation(y_true, y_probs, n_iterations=1000, random_state=42):
    """Runs 1,000 iterations of sampling with replacement to generate 95% CIs."""
    np.random.seed(random_state)
    n_size = len(y_true)
    pr_aucs = []
    
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_probs_arr = y_probs.values if isinstance(y_probs, pd.Series) else y_probs
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        y_true_boot = y_true_arr[indices]
        y_probs_boot = y_probs_arr[indices]
        
        # Skip iteration if random sample happens to draw zero fraud cases
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        score = average_precision_score(y_true_boot, y_probs_boot)
        pr_aucs.append(score)
        
    lower = np.percentile(pr_aucs, 2.5)
    upper = np.percentile(pr_aucs, 97.5)
    mean_score = np.mean(pr_aucs)
    
    return mean_score, lower, upper

def analyze_false_negatives(X_test_raw, y_true, y_preds):
    """Isolates the exact rows where the model failed to catch fraud."""
    fn_mask = (y_true == 1) & (y_preds == 0)
    return X_test_raw[fn_mask]