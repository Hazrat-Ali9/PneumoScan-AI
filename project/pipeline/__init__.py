from pathlib import Path
import sys

from project.components.data_ingestion import Data_Ingestion
from project.components.prepare_basemodel import Prepare_Segmentation_Model
from project.components.callbacks import Call_Backs
from project.components.model_trainer import Training
from project.components.model_evaluation import Evaluation
from project.configeration import Configeration_Manager
from project.exception import CustomException
from project.logger import logging


class Training_Pipeline:
    """
    Orchestrates the complete machine learning workflow including:
    - Data ingestion
    - Base model preparation
    - Callback creation
    - Model training
    - Model evaluation
    
    This class executes each pipeline step sequentially using configurations
    provided by the ConfigerationManager.
    """

    def __init__(self):
        """
        Initialize the TrainingPipeline with a single instance of
        ConfigerationManager to access all configuration sections.
        """
        self.config = Configeration_Manager()


    def run_data_ingestion(self):
        """
        Execute the data ingestion pipeline.
        
        Steps:
        - Download the dataset from the specified URL.
        - Extract the downloaded zip file.
        
        Raises:
            CustomException: If any part of ingestion fails.
        """
        try:
            logging.info(">>>>>>> Data Ingestion started <<<<<<<<<")
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = Data_Ingestion(data_ingestion_config)
            data_ingestion.download_data()
            data_ingestion.extract_zip_file()
            logging.info(">>>>>>> Data Ingestion completed <<<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)


    def run_prepare_base_model(self):
        """
        Execute the base model preparation pipeline.
        
        Steps:
        - Build the ResNet50 U-Net model architecture.
        - Compile the model with the specified loss and metrics.
        - Save the prepared model to disk.
        
        Raises:
            CustomException: If any part of model preparation fails.
        """
        try:
            logging.info(">>>>>>> Base Model Preparation started <<<<<<<<<")
            prepare_base_model_config = self.config.get_prepare_base_model_config()
            model_preparer = Prepare_Segmentation_Model(config=prepare_base_model_config)
            unet_model = model_preparer.build_resnet50_unet()
            model_preparer.save_model(
                path=Path(prepare_base_model_config.update_base_model), 
                model=unet_model
            )
            logging.info(">>>>>>> Base Model Preparation completed <<<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)
        

    def run_prepare_callbacks(self):
        """
        Execute the callback preparation pipeline.
        
        Steps:
        - Create TensorBoard and ModelCheckpoint callbacks based on configuration.
        - Compile a list of callbacks for use in model training.
        
        Raises:
            CustomException: If any part of callback preparation fails.
        """
        try:
            logging.info(">>>>>>> Callback Preparation started <<<<<<<<<")
            prepare_callback_config = self.config.get_prepare_callback_config()
            callbacks_manager = Call_Backs(config=prepare_callback_config)
            callbacks_list = callbacks_manager.get_callbacks()
            logging.info(">>>>>>> Callback Preparation completed <<<<<<<<<")
            return callbacks_list
        except Exception as e:
            raise CustomException(e, sys)
 
    
    def run_model_training(self, callbacks):
        """
        Execute the model training pipeline.
        
        Steps:
        - Load the prepared base model.
        - Create training and validation data generators.
        - Train the model using the specified callbacks.
        - Save the final trained model to disk.
        
        Note: This method assumes that the base model has already been prepared
        and that the data generators are properly set up to read from the ingested dataset.

        Args:
            callbacks (list): List of Keras callbacks to use during training."""
        
        try:
            logging.info(">>>>>>> Model Training started <<<<<<<<<")
            training_config = self.config.get_training_config()
            trainer = Training(config=training_config)
            trainer.get_base_model()
            train_data, val_data = trainer.train_valid_generator()
            trainer.train(train_data=train_data, val_data=val_data, callbacks=callbacks)
            logging.info(">>>>>>> Model Training completed <<<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)
        

    def run_model_evaluation(self):
        """
        Execute the model evaluation pipeline.
        
        Steps:
        - Load the trained model from disk.
        - Create a validation dataset generator.
        - Evaluate the model on the validation dataset.
        - Log evaluation results to MLflow and save scores to JSON.
        
        Raises:
            CustomException: If any part of model evaluation fails.
        """
        try:
            logging.info(">>>>>>> Model Evaluation started <<<<<<<<<")
            evaluation_config = self.config.get_model_evaluation_config()
            evaluator = Evaluation(config=evaluation_config)
            results = evaluator.evalution()
            # evaluator.log_mlflow(results=results)
            logging.info(">>>>>>> Model Evaluation completed <<<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)


    def run(self):
        """
        Execute the full ML pipeline in order:
        1. Data ingestion
        2. Base model preparation
        3. Model training
        4. Model evaluation
        
        Raises:
            CustomException: If any stage of the pipeline fails.
        """
        try:
            logging.info(">>>>>>> Training Pipeline started <<<<<<<<<")
            # self.run_data_ingestion()
            self.run_prepare_base_model()
            callbacks = self.run_prepare_callbacks()
            self.run_model_training(callbacks=callbacks)
            self.run_model_evaluation()
            logging.info(">>>>>>> Training Pipeline completed <<<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)


