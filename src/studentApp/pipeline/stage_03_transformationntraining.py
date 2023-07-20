from studentApp.config.configuration import ConfigurationManager
from studentApp.components.data_transformation import DataTransformation
from studentApp.components.training import Training
from studentApp import logger


STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transformation_data()
        X_train = data_transformation.X
        y_train = data_transformation.y
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.train(X_train, y_train)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

