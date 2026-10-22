# dashboard/app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os

# --- THE FIX ---
# Tell Python to look in the parent directory (project root) for the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---------------

st.set_page_config(page_title="FraudLens DI Dashboard", layout="wide")

# 1. Load the Phase 2 & Phase 3 Artifacts
@st.cache_resource
def load_pipeline():
    fe = joblib.load("artifacts/feature_engineer.joblib")
    model = joblib.load("artifacts/calibrated_xgb.joblib")
    return fe, model

# 2. Load a sample of the test set for simulation
@st.cache_data
def load_sample_data():
    X_test = pd.read_pickle("artifacts/X_test.pkl")
    y_test = pd.read_pickle("artifacts/y_test.pkl")
    
    # Grab 5 fraud cases and 15 legitimate cases to play with
    fraud_idx = y_test[y_test == 1].sample(5, random_state=42).index
    legit_idx = y_test[y_test == 0].sample(15, random_state=42).index
    
    sample_X = pd.concat([X_test.loc[fraud_idx], X_test.loc[legit_idx]])
    sample_y = pd.concat([y_test.loc[fraud_idx], y_test.loc[legit_idx]])
    
    # Shuffle
    sample_X['True_Class'] = sample_y
    return sample_X.sample(frac=1, random_state=42).reset_index(drop=True)

fe, model = load_pipeline()
OPTIMAL_THRESHOLD = 0.01  # From our Phase 4 Cost Matrix

st.title("🛡️ FraudLens: Decision Intelligence Engine")
st.markdown("""
This dashboard simulates real-time transaction scoring. The underlying XGBoost model has been calibrated via Isotonic Regression, and the decision threshold is strictly optimized to minimize the bank's dollar loss ($25 per review vs. $1,000 per missed fraud).
""")

df_sample = load_sample_data()

st.subheader("Live Transaction Feed")
selected_idx = st.selectbox("Select an incoming transaction to route:", df_sample.index)

if st.button("Score Transaction"):
    # 1. Extract raw data
    raw_row = df_sample.iloc[[selected_idx]].drop(columns=['True_Class'])
    true_label = df_sample.iloc[selected_idx]['True_Class']
    
    # 2. Pass through our leakage-proof feature engineer
    engineered_row = fe.transform(raw_row)
    
    # 3. Get calibrated probability
    prob = model.predict_proba(engineered_row)[0, 1]
    is_flagged = prob >= OPTIMAL_THRESHOLD
    
    # 4. Display Results
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Calibrated Fraud Probability", f"{prob * 100:.2f}%")
    col2.metric("Business Threshold", f"{OPTIMAL_THRESHOLD * 100:.2f}%")
    col3.metric("True Ground Reality", "FRAUD" if true_label == 1 else "LEGITIMATE")
    
    if is_flagged:
        st.error(f"🚨 **ALERT TRIGGERED:** Probability ({prob*100:.2f}%) exceeds the cost-optimized threshold ({OPTIMAL_THRESHOLD*100:.2f}%). Routing to human investigation team.")
    else:
        st.success(f"✅ **CLEARED:** Probability ({prob*100:.2f}%) is below the investigation threshold. Transaction approved.")
        
    st.markdown("### Engineered Features (Under the Hood)")
    st.write(engineered_row[['Amount_Log', 'Hour_Sin', 'Hour_Cos', 'Amount_Hourly_ZScore', 'PCA_Top5_Magnitude']])
