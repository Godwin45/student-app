from studentApp import logger
from studentApp.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from studentApp.pipeline.stage_02_feature_engineering import FeatureEngineeringPipeline
from studentApp.pipeline.stage_03_transformationntraining import DataTransformationPipeline



STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Feature Engineering stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   obj = FeatureEngineeringPipeline()
   obj.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e



STAGE_NAME = "Training"

try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
      obj = DataTransformationPipeline()
      obj.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e





