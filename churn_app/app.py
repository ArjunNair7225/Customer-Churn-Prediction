import streamlit as st
import joblib
import pandas as pd
import time

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Churn Risk Dashboard",
    layout="centered"
)

# ---------------- DARK PROFESSIONAL THEME ---------------- #
st.markdown("""
<style>

/* Entire App Background */
.stApp {
    background-color: #0F172A;
    color: white;
}

/* Main Card */
.block-container {
    padding: 2rem;
    background-color: #1E293B;
    border-radius: 15px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
}

/* Titles */
h1 {
    text-align: center;
    color: #F8FAFC;
}

h2, h3 {
    color: #CBD5E1;
}

/* Labels */
label {
    color: #E2E8F0 !important;
}

/* Button Styling */
.stButton > button {
    background-color: #3B82F6;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    font-size: 16px;
}

.stButton > button:hover {
    background-color: #2563EB;
    color: white;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    color: #38BDF8;
    font-size: 28px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("churn_model.pkl")

# ---------------- HEADER ---------------- #
st.title("Customer Churn Risk Analysis Dashboard")
st.caption("Machine Learning powered churn probability prediction system")

st.markdown("---")

# ---------------- USER INPUT SECTION ---------------- #
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 80)
    watch_hours = st.number_input("Total Watch Hours", 0.0)
    last_login_days = st.number_input("Days Since Last Login", 0)
    monthly_fee = st.number_input("Monthly Subscription Fee", 0.0)
    number_of_profiles = st.number_input("Number of Profiles", 1)

with col2:
    avg_watch_time_per_day = st.number_input("Avg Watch Time Per Day", 0.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    region = st.selectbox("Region", ["Asia", "Europe", "America"])
    device = st.selectbox("Device Used", ["Mobile", "TV", "Laptop"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI"])
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Drama", "Comedy", "Horror"])

inactive_flag = 1 if last_login_days > 14 else 0

st.markdown("---")

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict Churn Risk"):

    input_data = pd.DataFrame({
        "age": [age],
        "watch_hours": [watch_hours],
        "last_login_days": [last_login_days],
        "monthly_fee": [monthly_fee],
        "number_of_profiles": [number_of_profiles],
        "avg_watch_time_per_day": [avg_watch_time_per_day],
        "gender": [gender],
        "subscription_type": [subscription_type],
        "region": [region],
        "device": [device],
        "payment_method": [payment_method],
        "favorite_genre": [favorite_genre],
        "inactive_flag": [inactive_flag]
    })

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Churn Risk Score")

    progress = st.progress(0)
    for i in range(int(probability * 100) + 1):
        progress.progress(i)
        time.sleep(0.01)

    st.metric(
        label="Predicted Churn Probability",
        value=f"{probability*100:.2f}%"
    )

    st.markdown("---")

    if probability > 0.6:
        st.error("🔴 High Risk Customer – Immediate Retention Required")
    elif probability > 0.3:
        st.warning("🟡 Medium Risk Customer – Monitor Engagement")
    else:
        st.success("🟢 Low Risk Customer – Strong Engagement")

    st.info("Prediction based on Logistic Regression probability output.")