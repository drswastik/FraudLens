# src/fraudlens/data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_and_audit_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    
    # 1. Schema & Null Check
    required_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Dataset contains {null_count} missing values!")
        
    # 2. Remove Exact Duplicates (Prevents train-to-test memorization leakage)
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = initial_rows - len(df)
    
    fraud_count = df['Class'].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    
    print("=== DATA AUDIT REPORT ===")
    print(f"Total Valid Rows : {len(df):,} (Dropped {dropped:,} exact duplicates)")
    print(f"Legitimate Cases : {len(df) - fraud_count:,} ({100 - fraud_pct:.4f}%)")
    print(f"Fraud Cases      : {fraud_count:,} ({fraud_pct:.4f}%)")
    print(f"Imbalance Ratio  : 1 Fraud per {int((len(df) - fraud_count) / fraud_count)} Legitimate transactions")
    
    return df

def create_splits(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Creates a strict 70% Train / 15% Validation / 15% Test stratified split.
    """
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Isolate untouched test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Split remaining into train and validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    
    print("\n=== PARTITION SUMMARY ===")
    print(f"Train Set      : {len(X_train):7,} rows | {y_train.sum():3} frauds ({y_train.mean()*100:.4f}%)")
    print(f"Validation Set : {len(X_val):7,} rows | {y_val.sum():3} frauds ({y_val.mean()*100:.4f}%)")
    print(f"Test Set       : {len(X_test):7,} rows | {y_test.sum():3} frauds ({y_test.mean()*100:.4f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test