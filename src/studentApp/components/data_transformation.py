import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.train_df = None
        self.train = None
        self.X = None
        self.y = None
        self.preprocessor = self._create_preprocessor()

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

    def transformation_data(self):
        train_df = pd.read_csv(self.config.train_df)

        X = train_df.drop(columns=['math_score'], axis=1)
        y = train_df['math_score']

        self.X = self.preprocessor.fit_transform(X)
        self.y = y

        print("Data transformation complete")
