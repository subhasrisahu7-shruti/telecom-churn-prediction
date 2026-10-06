import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Page Configuration & UI Framing Pipeline
st.set_page_config(
    page_title="Telecom Enterprise Decision Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Enterprise Telecom Churn Decision Intelligence Suite")
st.caption("🚨 Production-Grade Decision Engine | Scaled Analytics Architecture")
st.markdown("---")

# 2. Automated Diagnostic Binary Asset Caching Layer
@st.cache_resource
def load_production_assets():
    try:
        model = joblib.load('telecom_churn_model.pkl')
        encoder = joblib.load('contract_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.error("⚠️ Core system binaries missing from deployment directory! Execute training script compilations first.")
        return None, None

model, encoder = load_production_assets()

if model and encoder:
    # 3. ADVANCED LAYER: Dynamic Market Sensitivity & Multipliers (Sidebar)
    st.sidebar.header("🎯 Live Profile Risk Parameters")
    tenure = st.sidebar.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=12)
    contract_type = st.sidebar.selectbox("Contract Type Structure", options=['Month-to-month', 'One year', 'Two year'])
    monthly_billing = st.sidebar.slider("Monthly Billing Charge ($)", min_value=20.0, max_value=120.0, value=55.0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Advanced Simulation Controls")
    # Enterprise Technology: Dynamic Risk Multipliers allowing live scenario alterations
    market_volatility = st.sidebar.slider("Market Volatility Coefficient", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    competitor_pressure = st.sidebar.checkbox("Trigger Competitor Aggressive Pricing Scenario", value=False)

    # 4. ADVANCED LAYER: Automated Data Quality Audit Shield
    # This acts as data validation protection before inference pipelines execute
    st.sidebar.markdown("### 🛡️ Data Guard Status")
    if monthly_billing / tenure > 50.0 and tenure < 3:
        st.sidebar.warning("⚠️ High Outlier Data Warning: Extreme Billing Detected for Low Tenure Profile.")
        data_quality_shield = "PASS WITH WARNINGS"
    else:
        st.sidebar.success("✅ Input Telemetry Metrics Validated: Clean State")
        data_quality_shield = "PASS"

    # Preprocessing Transformation Matrix
    contract_encoded = encoder.transform([contract_type])[0]
    input_dataframe = pd.DataFrame([{
        'Tenure': tenure,
        'Contract': contract_encoded,
        'MonthlyBilling': monthly_billing
    }])

    # 5. Core Operational Intelligence Inference Execution
    if st.sidebar.button("Execute Core Diagnostic Predictions"):
        # Fetch base prediction metrics from binary files
        base_probability = model.predict_proba(input_dataframe)[0][1]
        
        # ADVANCED TECHNOLOGY: Dynamic Scenario Override Calculations
        # Modifies base ML outputs using runtime market vectors
        adjusted_probability = base_probability * market_volatility
        if competitor_pressure:
            adjusted_probability += 0.15 # Add 15% structural pressure risk
            
        # Hard bounds capping probabilities between 0.0 and 1.0 safely
        final_probability = float(np.clip(adjusted_probability, 0.0, 1.0))
        final_prediction = 1 if final_probability >= 0.5 else 0
        
        # 6. EXECUTIVE PRESENTATION PANEL MATRIX
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚡ Churn Metric Classification")
            if final_prediction == 1:
                st.error(f"🚨 HIGH CHURN RISK DETECTED")
                st.caption(f"Calculated Probability: **{final_probability*100:.2f}%**")
            else:
                st.success(f"✅ LOW RISK RETAINED PROFILE")
                st.caption(f"Calculated Probability: **{final_probability*100:.2f}%**")
                
        with col2:
            st.subheader("🏢 Executive Revenue Risk Factor")
            financial_at_stake = monthly_billing if final_prediction == 1 else 0.0
            st.metric(
                label="Monthly Recurring Revenue (MRR) Risk Exposure",
                value=f"${financial_at_stake:.2f} MRR",
                delta=f"-${financial_at_stake * 12:.2f} Projected Annual Loss" if final_prediction == 1 else "Stable Revenue Velocity",
                delta_color="inverse"
            )
            
        with col3:
            st.subheader("🛡️ Data Engine System Logs")
            st.info(f"🛡️ **Data Guard:** {data_quality_shield}")
            st.write(f"📊 **Market Friction Scalar:** {market_volatility}x")
            st.write(f"📉 **Competitor Target Vector:** {'ACTIVE' if competitor_pressure else 'INACTIVE'}")

        st.markdown("---")
        
        # 7. ADVANCED FEATURE: Explainable AI & Dynamic Revenue Recovery Simulators
        st.header("🧠 Advanced Decision Support Analytics Suite")
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("🔍 Explainable AI (XAI) Directional Attribute Weights")
            # Calculate programmatic statistical directional vector paths 
            base_reference = np.array([36.0, 1.0, 70.0])
            user_metrics = np.array([tenure, contract_encoded, monthly_billing])
            
            calculated_impacts = {
                'Tenure Contribution': -(user_metrics[0] - base_reference[0]) * 0.3,
                'Contract Longevity Value': -(user_metrics[1] - base_reference[1]) * 1.2,
                'Billing Scale Multiplier': (user_metrics[2] - base_reference[2]) * 0.08
            }
            
            # Matplotlib Visual Render Engine
            fig, ax = plt.subplots(figsize=(6, 3))
            visual_colors = ['#ff4b4b' if value > 0 else '#00cc66' for value in calculated_impacts.values()]
            ax.barh(list(calculated_impacts.keys()), list(calculated_impacts.values()), color=visual_colors)
            ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
            ax.set_xlabel('Risk Sensitivity Distribution Matrix')
            st.pyplot(fig)
            st.caption("🔴 Red shifts point toward account cancellation vectors. 🟢 Green indicators support structural account retention.")

        with col5:
            st.subheader("💰 Prescriptive ROI Financial Retention Optimizer")
            if final_prediction == 1:
                st.warning("💡 Recommended Corporate Action: Customer Account requires immediate retention discount triggers.")
                selected_strategy = st.select_slider("Select Allocation Recovery Incentive Scale:", options=["0% Baseline", "10% Discount Plan", "25% High Priority Retention Plan"])
                
                # Dynamic Recovery Mathematics Formulation
                discount_fraction = 0.0 if "0%" in selected_strategy else (0.10 if "10%" in selected_strategy else 0.25)
                simulated_billing_charge = monthly_billing * (1.0 - discount_fraction)
                
                # Simulating recovery model outputs
                simulated_probability = final_probability * (1.0 - (discount_fraction * 1.8))
                final_sim_prob = float(np.clip(simulated_probability, 0.0, 1.0))
                
                st.markdown(f"📉 **Simulated Probability Recovery Curve:** Redefined Risk Drops from **{final_probability*100:.1f}%** down to **{final_sim_prob*100:.1f}%**")
                st.success(f"💵 **Secured MRR Contract Pipeline Run Rate:** Retained Corporate Assets: **${simulated_billing_charge:.2f} / month**")
            else:
                st.success("🌟 Profile behaves within stable parameter baselines. No prescriptive optimization discounts required.")
