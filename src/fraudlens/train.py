# src/fraudlens/train.py
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

def evaluate_imbalance_strategies(X_train, y_train):
    """
    Tests 3 different ways to handle extreme imbalance using 5-Fold CV.
    Crucial Interview Point: SMOTE is strictly inside the pipeline to prevent leakage.
    """
    print("Evaluating Imbalance Strategies via 5-Fold CV...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Strategy 1: Do nothing (Baseline)
    pipe_base = Pipeline([('clf', LogisticRegression(max_iter=1000, random_state=42))])
    
    # Strategy 2: Class Weights (Algorithmic penalty)
    pipe_weight = Pipeline([('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
    
    # Strategy 3: SMOTE (Synthetic oversampling)
    pipe_smote = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    strategies = {'Base LR': pipe_base, 'Weighted LR': pipe_weight, 'SMOTE + LR': pipe_smote}
    results = {}
    
    for name, pipeline in strategies.items():
        # scoring='average_precision' is the PR-AUC metric
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
        results[name] = np.mean(scores)
        print(f"  {name:15} | PR-AUC: {results[name]:.4f} (+/- {np.std(scores):.4f})")
        
    return results

def tune_xgboost(X_train, y_train, use_gpu=False):
    """
    Runs a Randomized Search across hyperparameter space using 5-Fold CV.
    This replaces our previous blind guessing of max_depth=6.
    """
    print("\nStarting XGBoost Hyperparameter Search (This will take some compute!)...")
    
    # Setup for RTX 2050 if use_gpu is True, else uses all CPU cores
    tree_method = 'hist'
    device = 'cuda' if use_gpu else 'cpu'
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method=tree_method,
        device=device,
        random_state=42
    )
    
    # The Grid: We search depth, learning rate, and tree subsampling to prevent overfitting
    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 10, 50, 100] # Let CV find the optimal class weight
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # RandomizedSearch tests 20 random combinations across 5 folds = 100 model trainings
    search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=20, 
        scoring='average_precision', 
        cv=cv, 
        verbose=1,
        n_jobs=-1, # Parallelize across folds
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    print(f"\nBest XGBoost PR-AUC : {search.best_score_:.4f}")
    print(f"Best Parameters     : {search.best_params_}")
    
    return search.best_estimator_