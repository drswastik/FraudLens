# src/fraudlens/features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe feature engineering transformer.
    Learns hourly spending statistics and scaling parameters strictly on the training fold,
    and applies them cleanly to validation/test folds.
    """
    def __init__(self):
        self.hourly_mean_amount_ = {}
        self.hourly_std_amount_ = {}
        self.global_mean_amount_ = 0.0
        self.global_std_amount_ = 1.0
        self.scaler_ = RobustScaler()
        self.cols_to_scale_ = []

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        
        # Extract 24-hour clock bucket (0 to 23) from elapsed seconds
        hour = ((df['Time'] // 3600) % 24).astype(int)
        df['Hour'] = hour
        
        # Learn average and std of transaction Amount per hour ONLY on training data
        stats = df.groupby('Hour')['Amount'].agg(['mean', 'std']).to_dict()
        self.hourly_mean_amount_ = stats['mean']
        self.hourly_std_amount_ = stats['std']
        
        self.global_mean_amount_ = df['Amount'].mean()
        self.global_std_amount_ = df['Amount'].std()
        
        # Fit RobustScaler (uses median/IQR so extreme fraud outliers don't distort scaling)
        engineered_df = self._create_features(X)
        self.cols_to_scale_ = [
            'Amount', 'Amount_Log', 'Amount_Hourly_ZScore', 
            'V_Interaction_14_17', 'V_Interaction_12_14', 'PCA_Top5_Magnitude'
        ]
        self.scaler_.fit(engineered_df[self.cols_to_scale_])
        return self

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Cyclical Time Encoding (Maps 24-hour clock onto a unit circle)
        hour_float = (df['Time'] / 3600.0) % 24.0
        df['Hour_Sin'] = np.sin(2 * np.pi * hour_float / 24.0)
        df['Hour_Cos'] = np.cos(2 * np.pi * hour_float / 24.0)
        
        # 2. Log-Transformed Amount (Fixes severe right-skewness)
        df['Amount_Log'] = np.log1p(df['Amount'])
        
        # 3. Micro-Transaction Flag (Fraudsters often test stolen cards with $0 - $1 charges)
        df['Is_Micro_Tx'] = (df['Amount'] <= 1.0).astype(float)
        
        # 4. Hourly Amount Z-Score (How unusual is this spend for this specific hour of the day?)
        hour_bucket = ((df['Time'] // 3600) % 24).astype(int)
        hr_mean = hour_bucket.map(self.hourly_mean_amount_).fillna(self.global_mean_amount_)
        hr_std = hour_bucket.map(self.hourly_std_amount_).fillna(self.global_std_amount_)
        df['Amount_Hourly_ZScore'] = (df['Amount'] - hr_mean) / (hr_std + 1e-5)
        
        # 5. High-Signal PCA Interactions (V14, V17, V12, V10, V16 have the strongest correlation with Fraud)
        df['V_Interaction_14_17'] = df['V14'] * df['V17']
        df['V_Interaction_12_14'] = df['V12'] * df['V14']
        df['PCA_Top5_Magnitude'] = np.sqrt(
            df['V14']**2 + df['V17']**2 + df['V12']**2 + df['V10']**2 + df['V16']**2
        )
        
        # Drop raw 'Time' so the model doesn't overfit to specific timestamps
        df = df.drop(columns=['Time'])
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._create_features(X)
        df[self.cols_to_scale_] = self.scaler_.transform(df[self.cols_to_scale_])
        return df