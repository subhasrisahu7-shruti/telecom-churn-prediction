# 📊 Enterprise Telecom Churn Decision Intelligence Suite

An end-to-end predictive machine learning pipeline and interactive decision-support application built during my online BCA program in AI & Data Science from Manipal University Jaipur (MUJ). This application bridges the gap between static predictive scripting and live corporate workflows.

## 🔗 Live Production Deployment
🎯 **Interactive Cloud App:** [https://streamlit.app](https://streamlit.app)

---

## 🚀 Key Operational Features
- **🎯 Live Profile Risk Evaluator:** Side-panel attribute sliders allowing retention managers to manually input standalone customer variables (Tenure, Contract, Monthly Billing) to compute real-time churn risk indicators instantly.
- **⚡ Database Scoring Engine:** Simulates automated extraction and processing of bulk telemetry records from an enterprise operational database.
- **🏢 Executive Financial Impact Summary:** Dynamically visualizes Monthly Recurring Revenue (MRR) at stake, tracking net recovered funds based on custom-tuned tactical concession discounts.
- **📊 Explanatory Risk Driver Visualizations:** Native horizontal feature importance plots derived directly from model weights to highlight principal churn indicators.

## 🧠 Model Technical Architecture
- **Algorithm Engine:** Random Forest Classifier (Scikit-Learn).
- **Performance Metric:** Benchmarked at a **70% to 81.00% predictive success rate**.
- **Data Engineering:** Continuous categorical handling using custom `LabelEncoder` scaling to evaluate high-risk month-to-month subscription structures.

## 🛠️ Technology Stack & Extensions
- **Core Languages:** Python (Pandas, NumPy)
- **Machine Learning:** Scikit-Learn
- **Dashboard UI Assembly:** Streamlit Web Framework
- **Cloud Infrastructure:** Streamlit Community Cloud Engine
- **Future Enterprise Roadmap:** Integration of PostgreSQL database nodes, XGBoost/LightGBM gradient boosting models, and Docker application container environments.

## 💻 Local Workspace Deployment Instructions
1. Clone the repository workspace:
   ```bash
   git clone https://github.com
   ```
2. Ingest the required production environment libraries:
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```
3. Boot up the local user interface script:
   ```bash
   streamlit run app.py
   ```
