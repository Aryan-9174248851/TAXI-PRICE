import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Get number of features model expects
try:
    n_features = model.n_features_in_
except:
    st.error("Model does not expose number of features")
    st.stop()

st.set_page_config(
    page_title="Taxi Price Prediction",
    page_icon="🚕"
)

st.title("🚕 Taxi Price Prediction")
st.write(f"This model expects **{n_features} features**")

st.divider()

# Dynamic input fields
inputs = []

for i in range(n_features):
    value = st.number_input(
        f"Feature {i+1}",
        value=0.0,
        step=0.1
    )
    inputs.append(value)

# Predict button
if st.button("Predict Fare 💰"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)

    st.success(f"🚖 Predicted Taxi Fare: ₹ {round(prediction[0], 2)}")

st.caption("Model-driven dynamic frontend")
