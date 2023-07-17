import pandas as pd
from sklearn.model_selection import train_test_split
import os
from studentApp import logger
from studentApp.entity.config_entity import FeatureConfig

class FeatureEngineering:
    def __init__(self, config:FeatureConfig):
        self.config = config
        self.student_df = None
        self.train = None
        self.test = None
    

    def preprocess_data(self):

        student_df = pd.read_csv(self.config.student_df)
     
        train,test = train_test_split(student_df, test_size=0.15)

        train_file_path = os.path.join("artifacts/data_ingestion", "train.csv")
        test_file_path = os.path.join("artifacts/data_ingestion", "test.csv")

        train.to_csv(train_file_path, index=False)
        test.to_csv(test_file_path, index=False)

        print("The training and testing CSV files have been saved.")
