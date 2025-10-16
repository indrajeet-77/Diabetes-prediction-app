import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
try:
    with open("logistic_regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error(
        "Model or scaler files not found. Please make sure 'logistic_regression_model.pkl' and 'scaler.pkl' are in the same directory."
    )
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the files: {e}")
    st.stop()


# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
)

# Minimal CSS - just for button styling
st.markdown(
    """
<style>
.stButton>button {
    background-color: #1E3A8A;
    color: white;
    border-radius: 10px;
    padding: 10px 24px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #1C3570;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- UI LAYOUT ---
st.title("🩺 Diabetes Prediction Model")
st.write(
    "This application uses a Logistic Regression model to predict the likelihood of a patient having diabetes based on their health metrics. Please enter the patient's details in the sidebar."
)


# Sidebar for user inputs
st.sidebar.header("Patient Health Metrics")
st.sidebar.write("Enter the values below:")


def user_input_features():
    """Creates sidebar inputs and returns them as a DataFrame."""
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 200, 117)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 23)
    insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 846, 30)
    bmi = st.sidebar.slider("BMI (weight in kg/(height in m)^2)", 0.0, 67.1, 32.0, 0.1)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider("Age (years)", 21, 81, 29)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Get user input
input_df = user_input_features()

# Display user input
st.subheader("Patient Input Parameters")
st.write(input_df)

# Prediction button
if st.button("Predict Diabetes Status"):
    # Scale the user input
    try:
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"The model predicts this patient has diabetes.")
            st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success(f"The model predicts this patient does not have diabetes.")
            st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

        st.info(
            "Disclaimer: This prediction is based on a machine learning model and is not a substitute for professional medical advice. Please consult a doctor for any health concerns."
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
