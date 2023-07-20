from studentApp.constants import *
import os
from pathlib import Path
from studentApp.utils.common import read_yaml, create_directories
from studentApp.entity.config_entity import (DataIngestionConfig,
                                               FeatureConfig,
                                               DataTransformationConfig,
                                               TrainingConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
      
    def get_feature_engineering_config(self) -> FeatureConfig:
        config = self.config.feature_engineering
        
        create_directories([config.root_dir])

        feature_eng_config = FeatureConfig(
            root_dir=Path(config.root_dir),
            student_df=Path(config.student_df),
          
        )

        return  feature_eng_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_df=Path(config.train_df),
          
        )

        return  data_transformation_config
    
    def get_training_config(self):
            training = self.config.training
            training_data_path = 'artifacts/data_ingestion/train.csv'  # Replace with the correct file path
            training_root_dir = 'artifacts'  # Replace with the desired root directory path

            create_directories([Path(training_root_dir)])

            training_config = TrainingConfig(
                root_dir=Path(training_root_dir),
                training_data=Path(training_data_path),
                trained_model_path=Path(training.trained_model_path),

                # Add other configuration attributes as needed
            )

            return training_config
