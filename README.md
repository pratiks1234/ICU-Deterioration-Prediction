# ICU Patient Deterioration Prediction System

This project builds an end-to-end clinical machine learning system to predict ICU patient deterioration using electronic health record time-series data. The system uses patient vitals, laboratory results, ICU stay information, and clinical trends to estimate deterioration risk over future time windows such as 6, 12, and 24 hours.

The project includes data preprocessing, rolling-window feature engineering, baseline machine learning models, advanced deep learning models, SHAP-based model explainability, and a Streamlit dashboard for clinician-facing risk visualization.

> Note: Raw clinical datasets such as MIMIC-IV/eICU are not included in this repository due to data-use restrictions.