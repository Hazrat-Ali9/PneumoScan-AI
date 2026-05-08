from project.entity.config import (Data_Ingestion_Config,
                                    Prepare_Basemodel_Config,
                                    Prepare_Callback_Config,
                                    Training_Config,
                                    Model_Evaluation_Config)
from project.utils import read_yaml, create_directories
from project.exception import CustomException
from project.logger import logging
from project.constants import *
from pathlib import Path
import sys
import os




class Configeration_Manager:
    """
    Manages the loading and parsing of configuration and parameter YAML files.
    Creates required directories and provides configuration objects 
    for different pipeline components.
    """

    def __init__(self, config_filepath: Path = CONFIG_YAML_FILE, param_filepath: Path = PARAM_YAML_FILE):
        """
        Initialize the Configuration Manager.

        Args:
            config_filepath (Path): Path object for the main configuration YAML.
            param_filepath (Path): Path object for the parameters YAML.

        Raises:
            CustomException: If YAML reading or directory creation fails.
        """
        try:
            self.config = read_yaml(Path(config_filepath))
            self.param = read_yaml(Path(param_filepath))

            # create_directories expects a LIST of paths
            create_directories([self.config.artifacts_root])
            
            logging.info("Configuration Manager initialized and artifacts root created.")

        except Exception as e:
            raise CustomException(e, sys)


    def get_data_ingestion_config(self) -> Data_Ingestion_Config:
        """
        Creates and returns the Data_Ingestion_Config object 
        by reading values from the config YAML file.

        Returns:
            Data_Ingestion_Config: Configuration object for data ingestion.

        Steps:
            - Extract data ingestion section from config YAML
            - Ensure root directory exists
            - Populate Data_Ingestion_Config dataclass with YAML values

        Raises:
            CustomException: If extraction or object creation fails.
        """
        try:
            config = self.config.data_ingestion

            # Create root directory for data ingestion
            create_directories([config.root_dir])

            return Data_Ingestion_Config(
                root_dir=config.root_dir,
                source_url=config.source_url,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def get_prepare_base_model_config(self)-> Prepare_Basemodel_Config:
        """
        Creates and returns the Prepare_Basemodel_Config object 
        by reading values from the config YAML file.

        Returns:
            Prepare_Basemodel_Config: Configuration object for base model preparation.

        Steps:
            - Extract prepare base model section from config YAML
            - Ensure root directory exists
            - Populate Prepare_Basemodel_Config dataclass with YAML values

        Raises:
            CustomException: If extraction or object creation fails.
        """
        try:
            config = self.config.prepare_base_model

            # Create root directory for base model preparation
            create_directories([config.root_dir])

            prepare_base_model_config = Prepare_Basemodel_Config(
            root_dir=config.root_dir,
            base_model=config.base_model,
            update_base_model=config.update_base_model,
            param_image_size=self.param.IMG_SIZE, 
            param_batch_size=self.param.BATCH_SIZE,
            param_epochs=self.param.EPOCHS, 
            param_learning_rate=self.param.LEARNING_RATE, 
            param_classics=self.param.CLASSICS, 
            param_weight=self.param.WEIGHTS, 
            param_include_top=self.param.INCLUDETOP
            
            )
        
            return prepare_base_model_config
        except Exception as e:
            raise CustomException(e, sys)


    def get_prepare_callback_config(self) -> Prepare_Callback_Config:
        """
        Creates and returns the Prepare_Callback_Config object 
        by reading values from the config YAML file.

        Returns:
            Prepare_Callback_Config: Configuration object for callbacks.
        Steps:
            - Extract prepare callback section from config YAML
            - Ensure root directory exists
            - Populate Prepare_Callback_Config dataclass with YAML values
        Raises:
            CustomException: If extraction or object creation fails.
        """
        try:
            config = self.config.prepare_callbacks
            # Extract folder path from full checkpoint filepath
            model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)

            # Ensure required directories exist
            create_directories([
                Path(model_ckpt_dir),
                Path(config.tensorboard_root_log_dir)
            ])


            prepare_callback_config = Prepare_Callback_Config(
                root_dir=config.root_dir,
                tensorboard_root_log_dir=config.tensorboard_root_log_dir,
                checkpoint_model_filepath=config.checkpoint_model_filepath
            )
        
            return prepare_callback_config
        except Exception as e:
            raise CustomException(e, sys)
    

    def get_training_config(self):
        """
        Create and return the training configuration for the model training step.

        This method reads values from the main configuration (`config.yaml`)
        and parameter settings (`params.yaml`), prepares the required directory 
        structure, and bundles all training-related settings into a `TrainingConfig` 
        object. 
        
        The returned `TrainingConfig` is later used by the Training Pipeline 
        to load the base model, set up data generators, apply augmentations, 
        and start the training process.

        Returns:
            TrainingConfig: A dataclass object containing all configuration
            required for model training, including:
                - root directory for training outputs  
                - final trained model save path  
                - updated base model file path  
                - training dataset directory  
                - image size, batch size, epochs  
                - augmentation flags  
                - learning rate  
        """
        training = self.config.training 
        prepare_base_model = self.config.prepare_base_model

        # Path to extracted dataset folder
        trainig_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "validation_dataset"
        )

        # Ensure training root directory exists
        create_directories([training.root_dir])

        # Build and return the TrainingConfig dataclass
        training_config = Training_Config(
            root_dir= training.root_dir, 
            trained_model_path= training.trained_model_path, 
            update_base_model= prepare_base_model.update_base_model, 
            training_data= trainig_data, 
            param_image_size= self.param.IMG_SIZE, 
            param_batch_size= self.param.BATCH_SIZE, 
            param_epochs= self.param.EPOCHS, 
            params_augmentation= self.param.AUGMENTATION,
            param_learning_rate= self.param.LEARNING_RATE
        )
        return training_config

    
    def get_model_evaluation_config(self) -> Model_Evaluation_Config:
        """
        Create and return the Model_Evaluation_Config dataclass.

        This method:
            1. Reads the model evaluation section from the main config.
            2. Ensures all required directories exist.
            3. Converts string paths to Path objects for OS-independent handling.
            4. Returns a fully populated Model_Evaluation_Config object.
        
        Returns:
            Model_Evaluation_Config: Dataclass containing paths, MLflow info, params, 
                                and evaluation-specific settings.
        """
        config = self.config.model_evaluation

        # Ensure directories exist
        create_directories([
            Path(config.root_dir),
            Path(config.scores_file_dir)
        ])

        # Build evaluation config
        model_evaluation_config = Model_Evaluation_Config(
            root_dir=Path(config.root_dir),
            report_file_dir=Path(config.report_file_dir),
            report_file_path=Path(config.report_file_path),
            report_file=config.report_file,
            threshold_accuracy=config.threshold_accuracy,
            scores_file_dir=Path(config.scores_file_dir),
            scores_file=config.scores_file,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
            mlflow_experiment_name=config.mlflow_experiment_name,
            all_params=self.param.to_dict(),
            param_image_size=self.param.IMG_SIZE,
            param_batch_size=self.param.BATCH_SIZE,
            training_data=Path(
                self.config.data_ingestion.unzip_dir
            ) / "validation_dataset"
        )

        return model_evaluation_config
      

        