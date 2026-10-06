import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Telecom Enterprise Decision Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Enterprise Telecom Churn Decision Intelligence Suite")
st.caption("Production-Grade Decision Engine | Integrated CSV Audit Logging")
st.markdown("---")

# 2. Automated Diagnostic Binary Asset Caching Layer
@st.cache_resource
def load_production_assets():
    try:
        model = joblib.load('telecom_churn_model.pkl')
        encoder = joblib.load('contract_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.error("⚠️ Core model binaries missing from current directory workspace! Execute training scripts first.")
        return None, None

model, encoder = load_production_assets()
LOG_FILE = "churn_history_log.csv"

def log_user_inference(tenure, contract, billing, final_prob, risk_status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = pd.DataFrame([{
        "Timestamp": timestamp,
        "Tenure_Months": tenure,
        "Contract_Structure": contract,
        "Monthly_Billing_USD": billing,
        "Calculated_Risk_Pct": round(final_prob * 100, 2),
        "Operational_Status": "HIGH RISK" if risk_status == 1 else "LOW RISK"
    }])
    if not os.path.isfile(LOG_FILE):
        new_record.to_csv(LOG_FILE, index=False)
    else:
        new_record.to_csv(LOG_FILE, mode='a', header=False, index=False)

if model and encoder:
    # 3. Sidebar UI Parameters Controller Layer
    st.sidebar.header("🎯 Live Profile Risk Parameters")
    tenure = st.sidebar.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=12)
    contract_type = st.sidebar.selectbox("Contract Type Structure", options=['Month-to-month', 'One year', 'Two year'])
    monthly_billing = st.sidebar.slider("Monthly Billing Charge ($)", min_value=20.0, max_value=120.0, value=55.0)
    
    contract_encoded = encoder.transform([contract_type])[0]
    
    input_dataframe = pd.DataFrame([{
        'Tenure': tenure,
        'Contract': contract_encoded,
        'MonthlyBilling': monthly_billing
    }])

    # 4. Core Operational Intelligence Inference Execution
    if st.sidebar.button("Execute Diagnostic Predictions"):
        # Calculate scores safely via loaded assets
        prob_scores = model.predict_proba(input_dataframe)
        final_probability = float(prob_scores[0][1])
        final_prediction = int(model.predict(input_dataframe)[0])
        
        # Save transaction entry record into CSV log database
        log_user_inference(tenure, contract_type, monthly_billing, final_probability, final_prediction)
        
        # Primary Metric Presentations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Churn Status Metric Result")
            if final_prediction == 1:
                st.error(f"🚨 HIGH CHURN RISK INDICATOR DETECTED ({final_probability * 100:.2f}%)")
            else:
                st.success(f"✅ LOW RISK RETAINED PROFILE ({final_probability * 100:.2f}%)")
                
        with col2:
            st.subheader("🏢 Executive Revenue Risk Factor")
            financial_at_stake = monthly_billing if final_prediction == 1 else 0.0
            st.metric(
                label="Monthly Recurring Revenue (MRR) Risk Exposure",
                value=f"${financial_at_stake:.2f} MRR",
                delta=f"-${financial_at_stake * 12:.2f} Projected Annual Loss" if final_prediction == 1 else "Stable Revenue Flow",
                delta_color="inverse"
            )

    # 5. Production Transaction Log Area
    st.markdown("---")
    st.header("📜 Historical Account Diagnostic Audit Database Trail")
    
    if os.path.isfile(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        st.dataframe(log_df.tail(8), use_container_width=True)
        
        csv_buffer = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Operational Audit Trail to CSV",
            data=csv_buffer,
            file_name=f"telecom_churn_audit_trail_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Telemetry database trail structural indexes initialized. Run a diagnostic calculation above to generate rows.")
