import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Model
# -------------------------------
try:
    model = pickle.load(open("fraud_model.pkl", "rb"))
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# Load Dataset
# -------------------------------
try:
    data = pd.read_csv("newdata.csv")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# -------------------------------
# UI
# -------------------------------
st.title("💳 Credit Card Fraud Detection System")
st.write("Detect whether a transaction is fraudulent or not")

st.info("Click below to test transactions")

# -------------------------------
# NORMAL PREDICTION
# -------------------------------
if st.button("🔍 Predict Transaction"):

    sample = data.sample(n=1)

    X_sample = sample.drop("Class", axis=1)
    actual = sample["Class"].values[0]

    prediction = model.predict(X_sample)[0]
    probability = model.predict_proba(X_sample)[0][1]

    st.subheader("📊 Result")

    if probability > 0.7:
        st.error("🚨 High Risk Fraud Transaction")
    elif probability > 0.3:
        st.warning("⚠️ Suspicious Transaction")
    else:
        st.success("✅ Normal Transaction")

    st.metric("Fraud Probability", f"{probability:.2e}")
    st.metric("Fraud Risk (%)", f"{probability*100:.6f}%")

    st.progress(min(float(probability), 1.0))

    if actual == 1:
        st.error("⚠️ This was actually a FRAUD transaction!")
    else:
        st.success("✔️ This was a NORMAL transaction")

    st.subheader("📄 Sample Transaction Data")
    st.dataframe(sample)

# -------------------------------
# FRAUD TEST BUTTON (SEPARATE)
# -------------------------------
if st.button("🧪 Test Fraud Case"):

    fraud = data[data["Class"] == 1].sample(n=1)

    X_sample = fraud.drop("Class", axis=1)

    prob = model.predict_proba(X_sample)[0][1]

    st.subheader("🚨 Fraud Test Result")

    st.error("⚠️ This is a FRAUD transaction")

    st.metric("Fraud Probability", f"{prob:.2e}")
    st.metric("Fraud Risk (%)", f"{prob*100:.6f}%")

    st.dataframe(fraud)