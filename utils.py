import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline

import joblib
import os




class DataLoader:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self, path="./data/diabetes_prediction_dataset.csv"):
        self.data = pd.read_csv(path)

    # Group smoking categories
    def __recategorize_smoking(self, smoking_status):
        if smoking_status in ["never", "No Info"]:
            return "non-smoker"
        elif smoking_status == "current":
            return "current"
        elif smoking_status in ["ever", "former", "not current"]:
            return "past_smoker"

    def prepare_data(self):
        # Remove duplicates
        duplicate_rows_data = self.data[self.data.duplicated()]
        print("number of duplicate rows: ", duplicate_rows_data.shape)
        self.data = self.data.drop_duplicates()

        # Remove 0.0187% values with 'Other' gender
        other_gender_ratio = (
            len(self.data[self.data["gender"] == "Other"]) * 100 / len(self.data)
        )
        print(f"Other gender: {other_gender_ratio}%")
        self.data = self.data[self.data["gender"] != "Other"]

        # Apply the function to the 'smoking_history' column
        self.data["smoking_history"] = self.data["smoking_history"].apply(
            self.__recategorize_smoking
        )

        self.X = self.data.drop("diabetes", axis=1)
        self.y = self.data["diabetes"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    [
                        "age",
                        "bmi",
                        "HbA1c_level",
                        "blood_glucose_level",
                        "hypertension",
                        "heart_disease",
                    ],
                ),
                ("cat", OneHotEncoder(), ["gender", "smoking_history"]),
            ]
        )
        self.X = self.preprocessor.fit_transform(self.X)
        feature_names = list(
            map(lambda x: x[5:], list(self.preprocessor.get_feature_names_out()))
        )
        self.X = pd.DataFrame(data=self.X, columns=feature_names)

        return self.data, self.X, self.y

    def transform_data(self, data):
        return self.preprocessor.transform(data)

    def save_preprocessor(self):
        directory = "models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(self.preprocessor, "./models/preprocessor.pkl")

    def create_model_pipeline(self, model, model_params):
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        clf = imbPipeline(
            steps=[
                ("over", over),
                ("under", under),
                ("classifier", model(**model_params)),
            ]
        )
        return clf

    def fit_model_pipeline(self, model_pipeline):
        if self.X_train is None:
            self.get_data_split()
        model_pipeline.fit(self.X_train, self.y_train)

    def evaluate_model_pipeline(self, model_pipeline):
        y_pred = model_pipeline.predict(self.X_test)
        print(
            "Model ROC Area Under Curve: ",
            roc_auc_score(self.y_test, model_pipeline.predict_proba(self.X_test)[:, 1]),
        )
        print("Model Accuracy: ", accuracy_score(self.y_test, y_pred))

    def get_data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=2023
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_model_pipeline(self, model_name, model_pipeline):
        # Export the trained model to a file
        directory = "models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(model_pipeline, f"./models/{model_name}.pkl")
