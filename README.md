# Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts customer churn
probability using Logistic Regression and deploys an interactive
dashboard using Streamlit.

## Project Overview

Customer churn is a major challenge for subscription-based businesses.
This project builds a predictive model to identify customers at high
risk of churn and provides probability-based risk segmentation for
business decision support.

Key Components: - Data preprocessing and cleaning - Feature
engineering - Logistic Regression model training - Model evaluation
(ROC-AUC, Precision, Recall, F1-score) - Streamlit interactive dashboard
for real-time prediction

Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Streamlit
-   Joblib

## Model Performance

-   ROC-AUC: ~0.96
-   Accuracy: ~90%
-   Probability-based churn risk scoring
-   Risk segmentation: High Risk Medium Risk Low Risk

## Features Used

-   Age
-   Watch Hours
-   Days Since Last Login
-   Monthly Fee
-   Number of Profiles
-   Average Watch Time per Day
-   Gender
-   Subscription Type
-   Region
-   Device Used
-   Payment Method
-   Favorite Genre
-   Inactivity Flag

## Project Structure

Customer-Churn-Prediction/ churn_app/ app.py churn_model.pkl main.ipynb
requirements.txt README.txt

## Installation & Usage

1.  Clone the repository
2.  Install dependencies using pip install -r requirements.txt
3.  Run the Streamlit app using: streamlit run app.py

## Business Impact

This model helps businesses: - Identify at-risk customers early -
Improve retention strategies - Optimize marketing spend - Make
data-driven decisions

## Future Improvements

-   Add feature importance visualization
-   Deploy on Streamlit Cloud
-   Add model explainability (SHAP)
-   Implement advanced models (Random Forest, XGBoost)
-   Add model monitoring

### Author

Arjun Nair Machine Learning & Data Science Enthusiast


