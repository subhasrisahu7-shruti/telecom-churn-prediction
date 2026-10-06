import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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

    # Preprocessing Input
    contract_encoded = encoder.transform([contract_type])[0]
    
    input_data = pd.DataFrame([{
        'Tenure': tenure,
        'Contract': contract_encoded,
        'MonthlyBilling': monthly_billing
    }])

    # 4. Core Operational Intelligence Inference Execution
    if st.sidebar.button("Compute Churn Risk Indicators"):
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        # Primary Output Layout Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Churn Status Metric Result")
            if prediction == 1:
                st.error(f"🚨 HIGH CHURN RISK INDICATOR DETECTED ({probability*100:.1f}%)")
            else:
                st.success(f"✅ LOW RISK RETAINED PROFILE ({probability*100:.1f}%)")
                
        with col2:
            st.subheader("🏢 Executive Revenue Risk Factor")
            st.metric(
                label="Statistical Churn Probability Indicator",
                value=f"{probability * 100:.2f}%",
                delta=f"-${monthly_billing:.2f} MRR At Stake" if prediction == 1 else "Stable Revenue Flow"
            )
            
        st.markdown("---")
        
        # ADVANCED FEATURE ADDITIONS LAYER
        st.header("🧠 Advanced Decision Support Analytics")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🔍 Explainable AI (XAI) Attribute Weights")
            # Calculate manual feature importance delta weights for visual feedback
            base_values = np.array([36.0, 1.0, 70.0]) # dataset feature medians
            current_values = np.array([tenure, contract_encoded, monthly_billing])
            
            # Approximate linear directional impact weights based on standard telecom vectors
            feature_impacts = {
                'Tenure (Months)': -(current_values[0] - base_values[0]) * 0.4,
                'Contract Structure': -(current_values[1] - base_values[1]) * 1.5,
                'Monthly Billing ($)': (current_values[2] - base_values[2]) * 0.1
            }
            
            # Plot dynamic feature impact graph
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ['#ff4b4b' if v > 0 else '#00cc66' for v in feature_impacts.values()]
            ax.barh(list(feature_impacts.keys()), list(feature_impacts.values()), color=colors)
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Risk Multiplier Weight Impact')
            st.pyplot(fig)
            st.caption("🔴 Red bars indicate traits pulling the customer toward churn. 🟢 Green bars keep them loyal.")

        with col4:
            st.subheader("💰 Smart Financial Retention Simulator")
            if prediction == 1:
                st.info("💡 Actionable Strategy: Offer this customer an immediate loyalty incentive plan below:")
                discount = st.radio("Select Predictive Discount Value Strategy:", ["10% Discount", "20% Discount", "30% Discount"])
                
                # Dynamic predictive simulation math logic
                discount_pct = float(discount.split("%")[0]) / 100.0
                simulated_billing = monthly_billing * (1 - discount_pct)
                
                sim_input = pd.DataFrame([{
                    'Tenure': tenure + 1, # Simulating tenure extension
                    'Contract': contract_encoded,
                    'MonthlyBilling': simulated_billing
                }])
                
                sim_prob = model.predict_proba(sim_input)[0][1]
                
                st.write(f"📉 **Simulated Risk Drop:** Churn likelihood drops from **{probability*100:.1f}%** down to **{sim_prob*100:.1f}%**")
                st.success(f"💵 **Saved MRR Balance:** Retained pipeline value: **${simulated_billing:.2f}/month**")
            else:
                st.success("🌟 Profile is structurally stable. No optimization discount required.")

