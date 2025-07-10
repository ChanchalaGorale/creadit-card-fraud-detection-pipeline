import streamlit as st
from utils.ingest_and_predict import predict_and_store
import asyncio

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below to check for fraud and log real-time data.")

# All 29 input features
feature_names = [
    f"V{i}" for i in range(1, 29)
] + ["Amount"]

user_input = {}

cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        value = st.number_input(
            label=feature,
            key=feature,
            value=0.0,
            format="%.5f"
        )
        user_input[feature] = value

if st.button("Predict & Ingest"):
    try:
        prediction = asyncio.run(predict_and_store(user_input))

        if prediction == 1:
            st.error("Fraudulent Transaction Detected!")
        else:
            st.success("Transaction Seems Legitimate.")
    except Exception as e:
        st.error(f"Error: {str(e)}")