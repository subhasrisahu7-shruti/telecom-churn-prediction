import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Telecom Decision Intelligence Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Enterprise Telecom Churn Decision Intelligence Suite")
st.markdown("---")

# 2. Safe Binary Loading Component Layer
@st.cache_resource
def load_assets():
    try:
        # Load the upgraded XGBoost model and categorical label encoder
        model = joblib.load('telecom_churn_model.pkl')
        encoder = joblib.load('contract_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.error("⚠️ Core model binaries missing! Please run your training pipeline script first.")
        return None, None

model, encoder = load_assets()

if model and encoder:
    # 3. Sidebar Profile Attribute Configuration Input Control
    st.sidebar.header("🎯 Live Profile Risk Evaluator")
    
    tenure = st.sidebar.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=12)
    contract_type = st.sidebar.selectbox("Contract Type Structure", options=['Month-to-month', 'One year', 'Two year'])
    monthly_billing = st.sidebar.slider("Monthly Billing Charge ($)", min_value=20.0, max_value=120.0, value=55.0)

    # 4. Live Scoring Data Preprocessing Pipeline
    # Transform input selection matching the exact string representation used during training
    contract_encoded = encoder.transform([contract_type])[0]
    
    # Create the structured feature matrix input block
    input_data = pd.DataFrame([{
        'Tenure': tenure,
        'Contract': contract_encoded,
        'MonthlyBilling': monthly_billing
    }])

    # 5. Core Operational Intelligence Inference Execution
    if st.sidebar.button("Compute Churn Risk Indicators"):
        # Calculate raw churn probability scores via XGBoost
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        # Display Metric Output Presentation Panels
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Churn Status Metric Result")
            if prediction == 1:
                st.error("🚨 HIGH CHURN RISK INDICATOR DETECTED")
            else:
                st.success("✅ LOW RISK RETAINED CUSTOMER PROFILE")
                
        with col2:
            st.subheader("🏢 Executive Revenue Risk Factor")
            st.metric(
                label="Statistical Churn Probability Indicator",
                value=f"{probability * 100:.2f}%",
                delta=f"-{monthly_billing:.2f} $ MRR At Stake" if prediction == 1 else "Stable Revenue Flow"
            )
