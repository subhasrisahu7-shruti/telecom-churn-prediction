import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 1. Page Configuration & UI Framing Pipeline
st.set_page_config(
    page_title="Telecom Enterprise Decision Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Enterprise Telecom Churn Decision Intelligence Suite")
st.caption("🚨 Production-Grade Decision Engine | Integrated HTML/CSS/JS Layers & Automated CSV Audit Logging")
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

# --- NEW AUTOMATED DATA LOGGER SYSTEM LAYER ---
LOG_FILE = "churn_history_log.csv"

def log_user_inference(tenure, contract, billing, volatility, final_prob, risk_status):
    """Automatically records execution telemetries into a local file database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = pd.DataFrame([{
        "Timestamp": timestamp,
        "Tenure_Months": tenure,
        "Contract_Structure": contract,
        "Monthly_Billing_USD": billing,
        "Market_Volatility": volatility,
        "Calculated_Risk_Pct": round(final_prob * 100, 2),
        "Operational_Status": "HIGH RISK" if risk_status == 1 else "LOW RISK"
    }])
    
    # Check if the database log file exists to manage structural headers dynamically
    if not os.path.isfile(LOG_FILE):
        new_record.to_csv(LOG_FILE, index=False)
    else:
        new_record.to_csv(LOG_FILE, mode='a', header=False, index=False)

if model and encoder:
    # 3. Sidebar Profile Attribute Configuration Input Control
    st.sidebar.header("🎯 Live Profile Risk Parameters")
    tenure = st.sidebar.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=12)
    contract_type = st.sidebar.selectbox("Contract Type Structure", options=['Month-to-month', 'One year', 'Two year'])
    monthly_billing = st.sidebar.slider("Monthly Billing Charge ($)", min_value=20.0, max_value=120.0, value=55.0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Advanced Simulation Controls")
    market_volatility = st.sidebar.slider("Market Volatility Coefficient", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    competitor_pressure = st.sidebar.checkbox("Trigger Competitor Aggressive Pricing Scenario", value=False)

    # Preprocessing Transformation Matrix
    contract_encoded = encoder.transform([contract_type])
    input_dataframe = pd.DataFrame([{
        'Tenure': tenure,
        'Contract': contract_encoded,
        'MonthlyBilling': monthly_billing
    }])

    # Native HTML5 & CSS3 Styling Embed Injection
    st.markdown("""
        <style>
            .enterprise-card {
                background: linear-gradient(135deg, #1e1e2f 0%, #252545 100%);
                border-radius: 12px;
                padding: 24px;
                border: 2px solid #4f46e5;
                box-shadow: 0 8px 32px 0 rgba(79, 70, 229, 0.2);
                color: #ffffff;
                transition: all 0.3s ease-in-out;
            }
            .enterprise-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px 0 rgba(79, 70, 229, 0.4);
                border-color: #6366f1;
            }
            .metric-header {
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                color: #a5b4fc;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                margin-bottom: 8px;
            }
            .metric-value {
                font-size: 2.2rem;
                font-weight: 700;
                color: #ffffff;
                margin: 0;
            }
            .metric-footer {
                font-size: 0.85rem;
                color: #f43f5e;
                margin-top: 12px;
                font-weight: 500;
            }
            .js-box {
                background-color: #111827;
                border-radius: 8px;
                padding: 12px;
                margin-top: 15px;
                border-left: 4px solid #10b981;
            }
        </style>
    """, unsafe_allow_html=True)

    # 4. Core Operational Intelligence Inference Execution
    if st.sidebar.button("Execute Core Diagnostic Predictions"):
        base_probability = model.predict_proba(input_dataframe)
        
        # Apply scenario logic overrides
        adjusted_probability = base_probability * market_volatility
        if competitor_pressure:
            adjusted_probability += 0.15
            
        final_probability = float(np.clip(adjusted_probability, 0.0, 1.0))
        final_prediction = 1 if final_probability >= 0.5 else 0
        
        # Calculate financial indicators
        financial_at_stake = monthly_billing if final_prediction == 1 else 0.0
        annual_loss_projection = financial_at_stake * 12.0

        # EXECUTE AUTOMATED TELEMETRY DATA LOGGING LOOP
        log_user_inference(tenure, contract_type, monthly_billing, market_volatility, final_probability, final_prediction)

        # HTML, CSS & JavaScript Hybrid Core Module Render
        st.markdown(f"""
            <div class="enterprise-card">
                <div class="metric-header">🏢 Premium Executive ROI Dashboard Component</div>
                <div class="metric-value">${financial_at_stake:.2f} <span style="font-size:1.2rem; color:#9ca3af;">MRR At Risk</span></div>
                <div class="metric-footer">🚨 Projected Impact Matrix: -${annual_loss_projection:.2f} Estimated Annual Loss</div>
                
                <!-- Native Client-Side JavaScript Live Action Engine -->
                <div class="js-box">
                    <p style="color:#10b981; margin:0 0 8px 0; font-size:0.85rem; font-weight:bold;">⚡ Real-Time Client-Side JavaScript Runtime Interpreter:</p>
                    <label style="color:#d1d5db; font-size:0.8rem;">Adjust Strategy Retention Discount Scale:</label>
                    <input type="range" id="jsDiscountSlider" min="0" max="50" value="20" style="width:100%; accent-color:#10b981;" oninput="calculateSavings({financial_at_stake})">
                    
                    <p style="margin:10px 0 0 0; font-size:0.9rem; color:#e5e7eb;">
                        💰 Immediate Revenue Recovered via JS Engine: 
                        <span id="jsSavingsDisplay" style="font-weight:bold; color:#10b981;">$0.00 / month</span>
                    </p>
                </div>
            </div>

            <script>
                function calculateSavings(mrrAtStake) {{
                    var discountVal = document.getElementById("jsDiscountSlider").value;
                    var savings = mrrAtStake * (1 - (discountVal / 100));
                    document.getElementById("jsSavingsDisplay").innerHTML = "$" + savings.toFixed(2) + " / month (At " + discountVal + "% Discount Plan)";
                }}
                calculateSavings({financial_at_stake});
            </script>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # 5. Diagnostic Charts Framework Panels
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Analytical Attribution Metrics")
            base_reference = np.array([36.0, 1.0, 70.0])
            user_metrics = np.array([tenure, contract_encoded, monthly_billing])
            
            calculated_impacts = {
                'Tenure Distribution Scale': -(user_metrics - base_reference) * 0.3,
                'Contract Value Horizon': -(user_metrics - base_reference) * 1.2,
                'Billing Scale Weight': (user_metrics - base_reference) * 0.08
            }
            
            fig, ax = plt.subplots(figsize=(6, 3.2))
            visual_colors = ['#ff4b4b' if value > 0 else '#00cc66' for value in calculated_impacts.values()]
            ax.barh(list(calculated_impacts.keys()), list(calculated_impacts.values()), color=visual_colors)
            ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
            st.pyplot(fig)
            st.caption("🔴 Pulls toward subscriber cancellation. 🟢 Supports structural account retention.")

        with col2:
            st.subheader("🛡️ Strategic Engine Parameters")
            st.write(f"📊 **System Logic Flag Status:** {'HIGH RISK PROFILE' if final_prediction == 1 else 'CLEAN AUDIT PROFILE'}")
            st.write(f"📉 **Total Predictive Risk Probability Score:** {final_probability * 100:.2f}%")
            st.write(f"🎛️ **Applied Environmental Friction Coefficient:** {market_volatility}x Scaling")

    # --- NEW ADVANCED GRID ELEMENT: PRODUCTION LOG HISTORY INTERFACE ---
    st.markdown("---")
    st.header("📜 Live Production Transaction History Logs")
    
    if os.path.isfile(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        
        # Render the file system log entries into a clean tabular data table
        st.dataframe(log_df.tail(10), use_container_width=True)
        
        # Multi-format executive extraction download options
        csv_buffer = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Operational Audit Trail to CSV",
            data=csv_buffer,
            file_name=f"telecom_churn_audit_trail_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Audit database trail initialized. Run your first simulation to generate live transaction entries.")
