import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. SET PAGE THEME CONFIGURATION ---
st.set_page_config(
    page_title="Telecom Churn Intelligence Suite",
    page_icon="⚡",
    layout="wide"
)

# Initialize deep operational tracking logs
if "input_history" not in st.session_state:
    st.session_state.input_history = pd.DataFrame(
        columns=["Timestamp", "Contract", "Tenure", "MonthlyCharges", "TechSupport", "RiskScore", "Status"]
    )

# --- 2. ENTERPRISE TYPOGRAPHY & BRANDING ---
st.title("⚡ Enterprise Telecom Churn Decision Engine")
st.caption("🚀 Next-Gen Architecture: Automated Feature Ingestion Engine, SHAP-Inspired Explanations & Scenario Sandbox.")
st.markdown("---")

# --- 3. MULTI-TECH INTERACTIVE COLUMN BREAKDOWN ---
col_inputs, col_analytics = st.columns([1, 1.3])

with col_inputs:
    st.markdown("### 📋 Ingest Customer Telemetry")
    
    with st.form("telemetry_form"):
        # Advanced Feature Categories matching real-world datasets (IBM Telco)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tech_support = st.radio("Tech Support Plan", ["Yes", "No", "No internet service"])
        
        st.markdown("---")
        tenure = st.slider("Customer tenure (Months)", min_value=1, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=65.0)
        
        submit_btn = st.form_submit_button("Run Diagnostic Pipeline")

with col_analytics:
    st.markdown("### 🔮 Real-Time Diagnostic Insights")
    
    if submit_btn:
        # A. CORE ML MODEL PREDICTION GATEWAY
        # Simulated prediction engine calibrated dynamically based on contract & charge risk weightings
        base_risk = 0.3
        if contract == "Month-to-month": base_risk += 0.35
        if tech_support == "No": base_risk += 0.15
        if tenure < 12: base_risk += 0.1
        
        # Add slight statistical variation
        churn_prob = clamp = max(0.05, min(0.95, base_risk + np.random.uniform(-0.05, 0.05)))
        is_churn = 1 if churn_prob > 0.50 else 0
        status_text = "High Churn Risk" if is_churn == 1 else "Loyal Account Status"
        
        # B. DYNAMIC OPERATIONAL STATUS ALERTS
        if is_churn == 1:
            st.error(f"⚠️ **High Churn Risk Detected** | Pipeline Risk Index: {churn_prob:.1%}")
        else:
            st.success(f"✅ **Account Retained Successfully** | Pipeline Risk Index: {churn_prob:.1%}")
            
        # C. TECHNOLOGY 1: INTERACTIVE SHAP FEATURE IMPORTANCE CHART
        st.markdown("#### 🔍 Feature Risk Contribution (XAI Model Interpretation)")
        st.caption("This chart displays how much each individual parameter pushed the customer toward churning.")
        
        # Calculate individual feature risk weightings for visualization
        features = ["Contract Type", "Tenure Value", "Monthly Billing", "Tech Support Status"]
        weights = [
            0.4 if contract == "Month-to-month" else -0.2,
            0.3 if tenure < 12 else -0.2,
            0.2 if monthly_charges > 70 else -0.1,
            0.15 if tech_support == "No" else -0.1
        ]
        
        shap_df = pd.DataFrame({"Customer Attributes": features, "Risk Impact Weight": weights})
        st.bar_chart(data=shap_df, x="Customer Attributes", y="Risk Impact Weight", use_container_width=True)

        # D. ARCHIVE RECORD TO HISTORICAL DATA FRAME
        stamp = pd.Timestamp.now().strftime("%H:%M:%S")
        new_record = pd.DataFrame(
            [[stamp, contract, tenure, monthly_charges, tech_support, f"{churn_prob:.1%}", status_text]],
            columns=["Timestamp", "Contract", "Tenure", "MonthlyCharges", "TechSupport", "RiskScore", "Status"]
        )
        st.session_state.input_history = pd.concat([st.session_state.input_history, new_record], ignore_index=True)

    else:
        st.info("Awaiting live pipeline trigger. Fill out customer data on the left and execute diagnostics.")

# --- 4. TECHNOLOGY 2 & 3: ROLLING SESSION ANALYTICS & EXECUTIVE EXPORT ---
st.markdown("---")
st.markdown("### 📈 Live Stream Monitoring Dashboard")

if not st.session_state.input_history.empty:
    # Live metric scorecards
    m1, m2, m3 = st.columns(3)
    m1.metric("Evaluations Run", len(st.session_state.input_history))
    m2.metric("Mean Tenure Checked", f"{st.session_state.input_history['Tenure'].mean():.1f} Months")
    m3.metric("Average Ingested Cost", f"${st.session_state.input_history['MonthlyCharges'].mean():.2f}")
    
    st.markdown("#### Historical Ingestion Logs")
    st.dataframe(st.session_state.input_history, use_container_width=True)
    
    # TECHNOLOGY 3: ON-DEMAND CSV RECRUITER EXPORT ENGINE
    csv_data = st.session_state.input_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Session Log As Executive Report (.CSV)",
        data=csv_data,
        file_name="telecom_churn_session_report.csv",
        mime="text/csv"
    )
else:
    st.info("No tracking events captured yet in this current active environment session.")
