import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Enterprise Churn Decision Engine", layout="wide")

# Clean, professional typography markers
st.title("📊 Next-Gen Telecom Churn Decision Intelligence Suite")
st.caption("Production Framework: Automated SQL Telemetry Ingestion Engine & Advanced Gradient Boosting Simulations.")

# --- 1. MODEL ARTIFACT LOADING ---
@st.cache_resource
def load_trained_artifacts():
    with open("telecom_churn_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    artifacts = load_trained_artifacts()
    prod_model = artifacts["classifier"]
    le_gender = artifacts["gender_encoder"]
    le_contract = artifacts["contract_encoder"]
    st.success("✅ Model Architecture Registry Connected: Random Forest Baseline Loaded!")
except FileNotFoundError:
    st.error("❌ 'telecom_churn_model.pkl' missing. Please ensure it is saved in your VS Code project directory.")
    st.stop()

# --- 2. ADVANCED INTERACTIVE SIDEBAR SIMULATOR ---
st.sidebar.header("🎯 Live Profile Risk Evaluator")
st.sidebar.markdown("Manually evaluate risk flags for single client settings.")

input_gender = st.sidebar.selectbox("Gender Group", options=["Female", "Male"])
input_contract = st.sidebar.selectbox("Subscription Architecture", options=["Month-to-month", "One year", "Two year"])
input_tenure = st.sidebar.slider("Client Tenure (Months)", min_value=1, max_value=72, value=8)
input_charges = st.sidebar.slider("Billing Profile (Monthly ₹)", min_value=20.0, max_value=150.0, value=85.0)

# Real-time inference compilation logic
single_profile = pd.DataFrame({
    "Gender": [input_gender],
    "Contract": [input_contract],
    "TenureMonths": [input_tenure],
    "MonthlyCharges": [input_charges]
})

encoded_profile = single_profile.copy()
encoded_profile["Gender"] = le_gender.transform(encoded_profile["Gender"])
encoded_profile["Contract"] = le_contract.transform(encoded_profile["Contract"])

single_prob = prod_model.predict_proba(encoded_profile)[0, 1] * 100
single_pred = prod_model.predict(encoded_profile)[0]

st.sidebar.markdown("---")
st.sidebar.markdown(f"#### Churn Probability Index: **{single_prob:.2f}%**")
if single_pred == 1:
    st.sidebar.error("🚨 System Status: High Risk (Issue Retention Offer)")
else:
    st.sidebar.success("🟢 System Status: Account Stable")

# --- 3. NEW FEATURE: RETENTION STRATEGY SIMULATOR TOOL ---
st.sidebar.markdown("---")
st.sidebar.header("💸 Retention Offer Simulator")
st.sidebar.markdown("Calculate proactive offer adjustments to win back high-risk accounts.")
discount_rate = st.sidebar.slider("Proposed Retention Discount (%)", min_value=5, max_value=50, value=20, step=5)


# --- 4. HIGH-TECH INTERFACE TABS ---
tab_dashboard, tab_tech_stack = st.tabs(["🚀 Operational Control Room", "🏗️ Architecture Stack Details"])

with tab_dashboard:
    # Set up fixed placeholder variables for initial dashboard layout state
    if 'batch_run_executed' not in st.session_state:
        st.session_state.batch_run_executed = False
        st.session_state.raw_database_df = pd.DataFrame()
        st.session_state.flagged_only = pd.DataFrame()

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### ⚡ Database Scoring Engine")
        st.markdown("Simulate extracting raw records from live operational databases.")
        
        if st.button("🔄 Execute Daily SQL Batch Scoring Run", type="primary"):
            np.random.seed(12)
            batch_size = 10
            
            raw_database_df = pd.DataFrame({
                "AccountID": [f"TEL-ID-{np.random.randint(7000, 9999)}" for _ in range(batch_size)],
                "Gender": np.random.choice(["Male", "Female"], size=batch_size),
                "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=batch_size, p=[0.6, 0.2, 0.2]),
                "TenureMonths": np.random.randint(1, 24, size=batch_size),
                "MonthlyCharges": np.random.uniform(35.0, 125.0, size=batch_size)
            })
            
            scoring_df = raw_database_df.copy()
            scoring_df["Gender"] = le_gender.transform(scoring_df["Gender"])
            scoring_df["Contract"] = le_contract.transform(scoring_df["Contract"])
            
            features = scoring_df[["Gender", "Contract", "TenureMonths", "MonthlyCharges"]]
            probs = prod_model.predict_proba(features)[:, 1]
            preds = prod_model.predict(features)
            
            raw_database_df["Risk Score (%)"] = np.round(probs * 100, 2)
            raw_database_df["Action Flag"] = np.where(preds == 1, "🔴 Flagged Risk", "🟢 Verified Safe")
            
            st.session_state.raw_database_df = raw_database_df
            st.session_state.flagged_only = raw_database_df[raw_database_df["Action Flag"] == "🔴 Flagged Risk"].sort_values(by="Risk Score (%)", ascending=False)
            st.session_state.batch_run_executed = True

        if st.session_state.batch_run_executed:
            st.markdown("#### 🎯 Critical Priority Retention List")
            if not st.session_state.flagged_only.empty:
                st.dataframe(st.session_state.flagged_only, use_container_width=True, hide_index=True)
            else:
                st.success("Database scanning cycle completed. No critical alert triggers found.")
        else:
            st.info("Trigger the batch execution engine button above to run the database pipeline model inferences.")
            
    with col_right:
        st.markdown("### 📊 System Risk Drivers")
        st.markdown("Relative impact weighting computed by the Random Forest algorithm:")
        
        feature_names = ["Gender", "Contract", "Tenure Months", "Monthly Charges"]
        importances = prod_model.feature_importances_
        
        chart_data = pd.DataFrame({
            "Features": feature_names,
            "Importance Weight": importances
        }).sort_values(by="Importance Weight", ascending=True)
        
        st.bar_chart(
            data=chart_data, x="Features", y="Importance Weight",
            color="#2e7d32", horizontal=True, use_container_width=True
        )

    # --- NEW FEATURE: STRATEGIC BUSINESS BANNER BLOCK ---
    if st.session_state.batch_run_executed and not st.session_state.flagged_only.empty:
        st.markdown("---")
        st.markdown("### 🏢 Executive Financial Impact Summary")
        
        # Performance matrix math calculations
        total_at_risk_revenue = st.session_state.flagged_only["MonthlyCharges"].sum()
        discount_fraction = discount_rate / 100.0
        predicted_retained_revenue = total_at_risk_revenue * (1.0 - discount_fraction)
        
        kpi_1, kpi_2, kpi_3 = st.columns(3)
        
        with kpi_1:
            st.metric(
                label="⚠️ Monthly Revenue At Risk", 
                value=f"₹{total_at_risk_revenue:.2f}",
                delta="Immediate Threat Logged",
                delta_color="inverse"
            )
            st.caption("Total monthly recurring charges of all accounts currently flagged with high churn risk indexes.")
            
        with kpi_2:
            st.metric(
                label="💸 Applied Retention Discount Cost", 
                value=f"₹{(total_at_risk_revenue * discount_fraction):.2f}",
                delta=f"Active Margin Impact (-{discount_rate}%)",
                delta_color="normal"
            )
            st.caption("The revenue concession cost required to save these accounts if they accept your retention discount.")
            
        with kpi_3:
            st.metric(
                label="🛡️ Net Recovered Monthly Revenue", 
                value=f"₹{predicted_retained_revenue:.2f}",
                delta="Saved Cash Flow Pipeline",
                delta_color="off"
            )
            st.caption("The secure subscription value safely retained inside your network cohort post-intervention.")

with tab_tech_stack:
    st.markdown("### 🛠️ Advanced Production Architecture Expansion Plan")
    st.markdown("This roadmap tracks the tools required to scale this script into a full enterprise data pipeline solution:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 🗄️ 1. Ingestion Layer")
        st.code("PostgreSQL / SQLite\nSQLAlchemy Python Wrapper\nAutomated Cron Job Pipelines", language="text")
        st.caption("Pulls telemetry data automatically from customer accounts database tables.")
    with col2:
        st.markdown("##### 🧠 2. Core Analytics Layer")
        st.code("XGBoost Classifier\nSHAP Model Interpretability\nScikit-Learn Scaling", language="text")
        st.caption("Replaces simple logic with gradient boosting trees for higher analytical precision.")
    with col3:
        st.markdown("##### 🌐 3. Deployment Layer")
        st.code("Streamlit Cloud Hosting\nGitHub Actions CI/CD\nDocker Container Engine", language="text")
        st.caption("Deploys the pipeline script online so cross-functional team leaders can access it via URL links.")
