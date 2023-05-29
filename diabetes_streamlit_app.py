import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils import DataLoader


# Initialize
@st.cache_data
def load_data():
    data_loader = DataLoader()
    data_loader.load_dataset()
    df, X, y = data_loader.prepare_data()
    return X


@st.cache_resource
def load_model():
    preprocessor = joblib.load("./models/preprocessor.pkl")
    model = joblib.load("./models/random_forest.pkl")
    return model, preprocessor


def map_values(value):
    if value == "Yes":
        return 1
    elif value == "No":
        return 0


# ex-AI LIME, SHAP
from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel


class AIExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.columns = list(
            map(lambda x: x[5:], list(preprocessor.get_feature_names_out()))
        )
        self.initialize_lime()
        self.initialize_shap()

    def initialize_lime(self):
        print(self.columns)
        self.lime = LimeTabular(
            self.model, self.X_train, feature_names=self.columns, random_state=1
        )

    def initialize_shap(self):
        background_val = pd.DataFrame(
            np.median(self.X_train, axis=0).reshape(1, -1), columns=self.columns
        )
        self.shap = ShapKernel(self.model, background_val)

    def get_lime_score_plot(self, input):
        lime_local = self.lime.explain_local(input, name="LIME")
        return lime_local.visualize(0)

    def get_shap_score_plot(self, input):
        shap_local = self.shap.explain_local(input, name="SHAP")
        return shap_local.visualize(0)
        # data = lime_local.data(0)
        # names = [*data['extra']['names'], *data['names']]
        # scores = [*data['extra']['scores'], *data['scores']]
        # return names, scores


st.set_page_config(layout="wide")

X_train = load_data()
loaded_model, preprocessor = load_model()
explainer = AIExplainer(loaded_model, X_train)

st.title("Diabetes Classification")
st.write(
    "This is a web app to predict the whether a patient is likely to have diabetes or not.\
        Several features are provided in the sidebar. Please adjust the\
        value of each feature. After that, click on the Predict button at the bottom to\
        see the prediction of the regressor."
)

with st.sidebar:
    gender = st.radio("Gender", ("Male", "Female"))
    smoking_history = st.radio(
        "Smoking History", ("non-smoker", "current", "past_smoker")
    )
    heart_disease_input = st.radio(
        "Suffering from Heart Disease?", ("Yes", "No"), index=1
    )
    heart_disease = map_values(heart_disease_input)
    hypertension_input = st.radio(
        "Suffering from Hypertension?", ("Yes", "No"), index=1
    )
    hypertension = map_values(hypertension_input)
    age = st.number_input(label="Age", min_value=0, max_value=2000, value=42, step=1)
    bmi = st.slider(label="BMI", min_value=10.0, max_value=100.0, value=27.3, step=0.1)
    HbA1c_level = st.slider(
        label="HbA1c Level", min_value=3.00, max_value=10.0, value=5.5, step=0.01
    )
    blood_glucose_level = st.slider(
        label="Blood Glucose Level",
        min_value=70.00,
        max_value=350.0,
        value=138.2,
        step=0.1,
    )

new_data = {
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "smoking_history": [smoking_history],
    "bmi": [bmi],
    "HbA1c_level": [HbA1c_level],
    "blood_glucose_level": [blood_glucose_level],
}

col1, col2 = st.columns((1, 2))

with col1:
    prButton = st.button("Predict")

# After predict is clicked show LIME, SHAP plots
if prButton:

    new_df = pd.DataFrame(data=new_data)
    inp = preprocessor.transform(new_df)
    prediction = loaded_model.predict(inp)

    prediction_text = "diabetic" if prediction else "not diabetic"
    st.write(f"Based on feature values, you are most likely **{prediction_text}**")
    
    st.title("LIME")
    st.plotly_chart(explainer.get_lime_score_plot(inp))
    st.title("SHAP")
    st.plotly_chart(explainer.get_shap_score_plot(inp))
