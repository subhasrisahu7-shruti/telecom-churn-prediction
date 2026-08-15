# Telecom Customer Churn Prediction (Classification)

An end-to-end Machine Learning project completed during my first year as an online BCA in AI & Data Science student at Manipal University Jaipur (MUJ). This project utilizes classification models to identify retail customers at high risk of canceling their subscriptions, enabling businesses to take proactive retention steps.

## 📊 Project Results
* **Model Used:** Random Forest Classifier
* **Overall Model Accuracy:** **81.00%** 🎯
* **Core Task:** Binary Classification (0 = Customer Stays, 1 = Customer Leaves)

## 📁 Repository Files
* `churn_model.py`: Main Python script containing the machine learning architecture.
* `requirements.txt`: Python package dependency list.

## 🛠️ Machine Learning Pipeline
1. **Data Preprocessing:** Standardized data fields, handled missing values locally, and utilized `LabelEncoder` to transform categorical data into machine-readable numeric formats.
2. **Feature Engineering:** Calculated logical data correlations between customer tenure and financial variables to predict retention trends.
3. **Model Training:** Implemented a `RandomForestClassifier` using Scikit-Learn to evaluate customer behavior branches.
4. **Evaluation:** Verified predictive performance, achieving an 81.00% success accuracy benchmark.
