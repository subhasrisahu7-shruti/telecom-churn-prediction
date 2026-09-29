import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. INITIALIZE USER INPUT TRACKING HISTORY ---
# This keeps track of what features users enter during their active session
if "input_history" not in st.session_state:
    st.session_state.input_history = pd.DataFrame(columns=["Tenure", "MonthlyCharges", "TotalCharges"])

# --- 2. APP HEADER ---
st.title("Enterprise Telecom Churn Decision Intelligence Suite")
st.write("Enter customer behavioral details below to predict the probability of churn.")

# --- 3. USER INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("📋 Customer Details")
    
    # Simple sliders and numeric inputs for your friends to test
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=10.0, max_value=8000.0, value=600.0)
    
    # Form submission button
    submit_button = st.form_submit_with_clicks = st.form_submit_button("Predict Churn & Track Data")

# --- 4. HANDLE SUBMISSION & METRIC TRACKING ---
if submit_button:
    # A. Save the user's current inputs into the session tracking log
    new_data = pd.DataFrame([[tenure, monthly_charges, total_charges]], 
                            columns=["Tenure", "MonthlyCharges", "TotalCharges"])
    st.session_state.input_history = pd.concat([st.session_state.input_history, new_data], ignore_index=True)
    
    # B. Run Churn Prediction (Placeholder logic for your model)
    # Replace this section with your actual model.predict() load sequence if needed
    st.subheader("🔮 Model Prediction Result")
    churn_probability = np.random.uniform(0.1, 0.9)  # Placeholder
    
    if churn_probability > 0.5:
        st.error(f"⚠️ High Risk: This customer is **likely to churn** (Probability: {churn_probability:.2%})")
    else:
        st.success(f"✅ Safe: This customer is **likely to stay** (Probability: {churn_probability:.2%})")

# --- 5. REAL-TIME REAL USER FEEDBACK DASHBOARD ---
st.markdown("---")
st.header("📊 Real-Time Interaction Dashboard")
st.write("This section tracks user behavior data across the current testing session.")

if not st.session_state.input_history.empty:
    # Calculate rolling averages from total user input submissions
    avg_tenure = st.session_state.input_history["Tenure"].mean()
    avg_monthly = st.session_state.input_history["MonthlyCharges"].mean()
    total_tests = len(st.session_state.input_history)
    
    # Display clear, responsive visual scorecard anchors
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Tests Conducted", value=int(total_tests))
    col2.metric(label="Avg Tenure Tested", value=f"{avg_tenure:.1f} Mo")
    col3.metric(label="Avg Monthly Charges", value=f"${avg_monthly:.2f}")
    
    # Let users inspect the raw background tracking dataset
    with st.expander("🔍 Click to view interactive input log history"):
        st.dataframe(st.session_state.input_history)
else:
    st.info("No interactive sessions logged yet. Adjust the settings above and click predict to begin tracking.")
