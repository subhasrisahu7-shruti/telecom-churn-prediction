import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 1. Page Configuration & Professional Framing
st.set_page_config(
    page_title="TelcoNexus | Enterprise Churn Decision Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED WEB DESIGN INJECTION: FULL SAAS CSS INTERFACE ENGINE ---
st.markdown("""
    <style>
        /* Hide default Streamlit clutter to present a clean custom web page */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Master Web Page Background styling */
        .stApp {
            background-color: #0b0f19 !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        }

        /* Glassmorphism Sticky Web Header Navigation Bar */
        .navbar-brand {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 40px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            position: sticky;
            top: 0;
            z-index: 999;
            margin-bottom: 25px;
            border-radius: 8px;
        }
        .nav-logo {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .nav-badge {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        /* Enterprise Web Hero Banner Layout Section */
        .hero-banner {
            background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.12), transparent), #111827;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            color: #9ca3af;
            font-size: 1rem;
            max-width: 800px;
            line-height: 1.6;
        }

        /* Advanced Dynamic SaaS Dashboard Layout Grid Cards */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .saas-card {
            background: #151f32;
            border-radius: 14px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .saas-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(99, 102, 241, 0.1);
        }
        .card-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .card-value {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
        }
        
        /* Client-Side JavaScript Interactive Playground Sandbox Box */
        .js-runtime-wrapper {
            background: #0f172a;
            border-radius: 10px;
            padding: 16px;
            margin-top: 20px;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        
        /* Styled Input range for Javascript web controls */
        .web-slider {
            width: 100%;
            height: 6px;
            background: #334155;
            border-radius: 9999px;
            accent-color: #10b981;
            margin: 12px 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- WEB PAGE NAVIGATION HEADER ---
st.markdown("""
    <div class="navbar-brand">
        <div class="nav-logo">⚡ TelcoNexus Platform</div>
        <div class="nav-badge">V2.4 Enterprise Web Release</div>
    </div>
""", unsafe_allow_html=True)

# --- HERO SECTION BANNER ---
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Predictive Revenue & Retention Operations Command</div>
        <div class="hero-subtitle">
            Welcome to the future of corporate subscriber management. This advanced web application integrates state-of-the-art 
            Machine Learning classifiers with real-time financial tracking, prescriptive JavaScript analytics models, 
            and automated transactional record caching. Use the live sidebar parameter inputs to execute profile checks.
        </div>
    </div>
""", unsafe_allow_html=True)

# 2. Production Diagnostic Asset Registry Layer
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

def log_user_inference(tenure, contract, billing, volatility, final_prob, risk_status):
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
    if not os.path.isfile(LOG_FILE):
        new_record.to_csv(LOG_FILE, index=False)
    else:
        new_record.to_csv(LOG_FILE, mode='a', header=False, index=False)

if model and encoder:
    # 3. Sidebar UI Parameters Controller Layer
    st.sidebar.header("🎯 Accounts Telemetry Feeds")
    tenure = st.sidebar.slider("Customer Tenure Length (Months)", min_value=1, max_value=72, value=12)
    contract_type = st.sidebar.selectbox("Contractual Agreement Type", options=['Month-to-month', 'One year', 'Two year'])
    monthly_billing = st.sidebar.slider("Subscriber Monthly Charges ($)", min_value=20.0, max_value=120.0, value=55.0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Simulation Modifiers")
    market_volatility = st.sidebar.slider("Market Churn Volatility Index", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    competitor_pressure = st.sidebar.checkbox("Apply High Aggressive Competitor Scenario", value=False)

    contract_encoded = encoder.transform([contract_type])
    input_dataframe = pd.DataFrame([{
        'Tenure': tenure,
        'Contract': contract_encoded,
        'MonthlyBilling': monthly_billing
    }])

    # 4. Core Diagnostic Evaluation Core Execution Loop
    if st.sidebar.button("Run Web Diagnostics Engine"):
        base_probability = model.predict_proba(input_dataframe)
        
        # Apply runtime volatility vector modifications
        adjusted_probability = base_probability * market_volatility
        if competitor_pressure:
            adjusted_probability += 0.15
            
        final_probability = float(np.clip(adjusted_probability, 0.0, 1.0))
        final_prediction = 1 if final_probability >= 0.5 else 0
        
        financial_at_stake = monthly_billing if final_prediction == 1 else 0.0
        annual_loss_projection = financial_at_stake * 12.0

        # Save transaction entry record into CSV database files
        log_user_inference(tenure, contract_type, monthly_billing, market_volatility, final_probability, final_prediction)

        # --- ADVANCED WEB COMPONENT LAYER: SAAS METRIC GRID GRID LAYOUT WITH EMBEDDED JAVASCRIPT ---
        # Constructs fully custom HTML/CSS Grid blocks containing client side JS execution loops
        st.markdown(f"""
            <div class="grid-container">
                <div class="saas-card" style="border-left: 4px solid {'#ef4444' if final_prediction == 1 else '#10b981'};">
                    <div class="card-label">⚡ Real-Time Churn Analysis Matrix</div>
                    <div class="card-value">{"🚨 HIGH CHURN RISK" if final_prediction == 1 else "✅ RETAINED PROFILE"}</div>
                    <p style="margin:8px 0 0 0; color:#9ca3af; font-size:0.9rem;">Computed Probability Index Score: <b>{final_probability*100:.2f}%</b></p>
                </div>
                <div class="saas-card" style="border-left: 4px solid #3b82f6;">
                    <div class="card-label">🏢 Operational Revenue Impact (MRR)</div>
                    <div class="card-value">${financial_at_stake:.2f} <span style="font-size:1rem; color:#9ca3af; font-weight:normal;">/ mo</span></div>
                    <p style="margin:8px 0 0 0; color:#ef4444; font-size:0.9rem;"><b>Projected Annualized Value Drag: -${annual_loss_projection:.2f} ARR</b></p>
                </div>
            </div>

            <!-- Full-Stack Web Component Layer: Responsive Prescriptive JavaScript Calculator Asset Module -->
            <div class="saas-card" style="margin-bottom:30px; border-top: 2px solid #10b981;">
                <div class="card-label" style="color:#10b981;">💰 Client-Side JavaScript Optimization Engine Playground</div>
Simulate customer loyalty incentives. Adjust the browser slider widget below to run client-side vector calculation queries without stressing server hardware routines:

Configure Strategy Discount Scale Percentage:
🛡️ Redeemed Monthly Contract Value: $0.00
📈 Net Financial Churn Mitigation Effectivity Enabled


function runWebCalculator(mrrValue) {{
var sliderPct = document.getElementById("webJsSlider").value;
var dynamicSavings = mrrValue * (1 - (sliderPct / 100));
document.getElementById("webSavings").innerHTML = "$" + dynamicSavings.toFixed(2) + " / month (At " + sliderPct + "% Strategic Cut)";
}}
runWebCalculator({financial_at_stake});

""", unsafe_allow_html=True)
st.markdown("---")
# 5. Advanced Model Interpretability Visual Graphics Layer
st.subheader("🔍 Production Model Graph Analysis & Feature Attribution Matrix")
col1, col2 = st.columns([3, 2])
with col1:
base_ref = np.array([36.0, 1.0, 70.0])
user_ref = np.array([tenure, contract_encoded, monthly_billing])
calculated_impacts = {
'Customer Account Longevity (Tenure)': -(user_ref - base_ref) * 0.3,
'Contract Structural Stability': -(user_ref - base_ref) * 1.2,
'Monthly Billing Weight Vector': (user_ref - base_ref) * 0.08
}
# Custom styled dark-theme Matplotlib figure to blend with the web page aesthetic
fig, ax = plt.subplots(figsize=(7, 2.8), facecolor='#151f32')
ax.set_facecolor('#151f32')
bar_colors = ['#ef4444' if val > 0 else '#10b981' for val in calculated_impacts.values()]
bars = ax.barh(list(calculated_impacts.keys()), list(calculated_impacts.values()), color=bar_colors, height=0.5)
ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
# Style text features to match web page design
ax.tick_params(colors='white', labelsize=9)
ax.xaxis.label.set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#334155')
ax.spines['bottom'].set_color('#334155')
ax.set_xlabel('Predictive Logic Push Scale', fontsize=9)
st.pyplot(fig)
with col2:
st.markdown(f"""

⚙️ Algorithmic Infrastructure Status

Core Model Protocol: Optimized Classifier Matrix
Dynamic Friction Bias: {market_volatility}x Modifier State
Competitor Pressure Mapping: {'ACTIVE DETECTION' if competitor_pressure else 'INACTIVE STATE'}
Hardware Processing State: Asynchronous Pipeline Loop


""", unsafe_allow_html=True)
# --- 6. PRODUCTION LOG GRID COMPONENT WEB AREA ---
st.markdown("---")
st.markdown('📜 Historical Account Diagnostic Audit Database Trail', unsafe_allow_html=True)
if os.path.isfile(LOG_FILE):
log_df = pd.read_csv(LOG_FILE)
st.dataframe(log_df.tail(8), use_container_width=True)
csv_buffer = log_df.to_csv(index=False).encode('utf-8')
st.download_button(
label="📥 Export Full Enterprise Audit Ledger to CSV",
data=csv_buffer,
file_name=f"telconexus_audit_ledger_{datetime.now().strftime('%Y%m%d')}.csv",
mime="text/csv"
)
else:
st.info("ℹ️ Telemetry database trail structural indexes initialized. Run a simulation computation step to generate rows.")

---


