import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("ev_model.pkl")

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
    st.write("DEBUG mileage:", mileage)
    st.write("DEBUG battery:", battery_status)

    input_df = pd.DataFrame([[mileage, battery_num]], columns=['Mileage','Battery_Status_Num'])

    # Maintenance Prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # Probability of Maintenance Needed

    # 🔴 MANUAL MILEAGE RULE (IMPORTANT)
    st.subheader("🔧 Maintenance Prediction")
    st.write("DEBUG mileage:", mileage)
    st.write("DEBUG battery:", battery_status)
    
    if mileage <= 30000 and battery_status == "New":
        st.success("Maintenance Not Needed")
    elif mileage >= 70000:
        st.error("Maintenance Needed (High Mileage)")
    elif pred == 1:
        st.error("Maintenance Needed")
    else:
        st.success("Maintenance Not Needed")
    
    st.subheader("🚨 Failure Risk")
    if mileage <= 30000 and battery_status == "New":
        st.success("LOW Risk")
    elif mileage >= 90000:
        st.error("HIGH Risk")
    elif prob < 0.4:
        st.success("LOW Risk")
    elif prob < 0.7:
        st.warning("MEDIUM Risk")
    else:
        st.error("HIGH Risk")

    # Battery Prediction
    st.subheader("🔋 Battery Prediction")
    if battery_status == "New":
        st.success("Battery OK")
    elif battery_status == "Weak":
        st.warning("Battery Weak")
    else:
        st.error("Battery Critical / Replace Soon")

   






