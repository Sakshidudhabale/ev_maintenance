import streamlit as st
import pandas as pd
import os
import pickle

# Load trained model
# Load trained EV model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ev_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("🚗 Smart EV Health Prediction System")

st.markdown("Predict **Maintenance**, **Battery Health**, and **Failure Risk** for your vehicle!")

# -------------------------------
# Inputs
# -------------------------------
mileage = st.number_input("Mileage (km)", min_value=0, value=30000)
battery_status = st.selectbox("Battery Status", ["New", "Weak", "Old"])

# Encode Battery_Status
battery_map = {"New": 0, "Weak": 1, "Old": 2}
battery_num = battery_map[battery_status]

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict EV Health"):
    input_df = pd.DataFrame([[mileage, battery_num]], columns=['Mileage','Battery_Status_Num'])

    # Maintenance Prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # Probability of Maintenance Needed

    st.subheader("🔧 Maintenance Prediction")
    if pred == 1:
        st.error("Maintenance Needed")
    else:
        st.success("Maintenance Not Needed")

    # Battery Prediction
    st.subheader("🔋 Battery Prediction")
    if battery_status == "New":
        st.success("Battery OK")
    elif battery_status == "Weak":
        st.warning("Battery Weak")
    else:
        st.error("Battery Critical / Replace Soon")

    # Failure Risk Prediction
    st.subheader("🚨 Failure Risk")
    if prob < 0.3:
        st.success("LOW Risk")
    elif prob < 0.6:
        st.warning("MEDIUM Risk")
    else:
        st.error("HIGH Risk")




