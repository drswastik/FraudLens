# src/fraudlens/evaluate.py
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, average_precision_score
import xgboost as xgb

def calibrate_model(best_model, X_train, y_train, X_val, y_val):
    print("Evaluating Uncalibrated Model...")
    # Get probabilities from our Phase 2 XGBoost model
    uncal_probs = best_model.predict_proba(X_val)[:, 1]
    
    # Brier score measures how far off the probabilities are from reality (0 or 1)
    uncal_brier = brier_score_loss(y_val, uncal_probs)
    uncal_pr = average_precision_score(y_val, uncal_probs)

    print("Calibrating Model using Isotonic Regression (5-Fold CV)...")
    # Extract the exact parameters our RandomizedSearch found in Phase 2
    params = best_model.get_params()
    base_xgb = xgb.XGBClassifier(**params)
    
    # Wrap it in a Scikit-Learn Calibrator
    calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, cv=5, method='isotonic')
    calibrated_xgb.fit(X_train, y_train)
    
    cal_probs = calibrated_xgb.predict_proba(X_val)[:, 1]
    cal_brier = brier_score_loss(y_val, cal_probs)
    cal_pr = average_precision_score(y_val, cal_probs)
    
    print(f"\n=== CALIBRATION RESULTS (Validation Set) ===")
    print(f"Uncalibrated Brier Score : {uncal_brier:.6f} (PR-AUC: {uncal_pr:.4f})")
    print(f"Calibrated Brier Score   : {cal_brier:.6f} (PR-AUC: {cal_pr:.4f})")
    print(f"Brier Score Improvement  : {((uncal_brier - cal_brier) / uncal_brier) * 100:.1f}%")
    
    return calibrated_xgb