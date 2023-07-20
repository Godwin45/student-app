import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor = self._create_preprocessor()
        self.model = self._load_model()

    def _create_preprocessor(self):
        numerical_columns = ["writing_score", "reading_score"]
        categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ]
        )

        return preprocessor

    def _load_model(self):
        return joblib.load(self.model_path)

    def fit_preprocessor(self, X):
        self.preprocessor.fit(X)

    def predict(self, features):
        # Check if the preprocessor is fitted before transforming
        if not hasattr(self.preprocessor, "transformers_"):
            raise ValueError("Preprocessor is not fitted. Call 'fit_preprocessor' with training data before making predictions.")

        data_scaled = self.preprocessor.transform(features)
        preds = self.model.predict(data_scaled)
        return preds


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        custom_data_input_dict = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }

        return pd.DataFrame(custom_data_input_dict)


