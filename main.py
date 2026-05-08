from project.pipeline import Training_Pipeline 
from project.exception import CustomException
import sys


if __name__ == "__main__":
    try:
        training_pipeline = Training_Pipeline()
        training_pipeline.run()
    except Exception as e:
        raise CustomException(e, sys)