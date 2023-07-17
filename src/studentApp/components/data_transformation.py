import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
from studentApp import logger
from studentApp.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        self.train_df = None
        self.train = None
    

    def transformation_data(self):

        train_df = pd.read_csv(self.config.train_df)

        X = train_df.drop(columns=['math_score'],axis=1)
        y = train_df['math_score']

        # Create Column Transformer with 3 types of transformers
        num_features = X.select_dtypes(exclude="object").columns
        cat_features = X.select_dtypes(include="object").columns

        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder()

        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", numeric_transformer, num_features),        
            ]
        )
        X = preprocessor.fit_transform(X)


        print("Data transformation complete")
