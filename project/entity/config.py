from dataclasses import dataclass 
from pathlib import Path 

@dataclass(frozen=True)
class Data_Ingestion_Config:
    """
    Dataclass for storing the configuration required to download and extract data.

    Attributes:
        root_dir (Path): Base directory for data artifacts.
        source_url (str): URL from which to download the data.
        local_data_file (Path): Local path where the data will be saved.
        unzip_dir (Path): Directory where the data will be extracted.
    """
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class Prepare_Basemodel_Config:
    """
    Dataclass for storing the configuration required to prepare the base model.

    Attributes:
        root_dir (Path): Base directory for base model artifacts.
        base_model (Path): Path to the base model file.
        update_base_model (Path): Path to the updated base model file.
        param_image_size (list): List of image sizes for training.
        param_batch_size (int): Batch size for training.
        param_epochs (int): Number of epochs for training.
    """
    root_dir: Path
    base_model: Path
    update_base_model: Path 
    param_image_size: list
    param_batch_size: int
    param_epochs: int 
    param_learning_rate: float
    param_classics: int
    param_weight:str
    param_include_top: bool


@dataclass(frozen=True)
class Prepare_Callback_Config:
    """
    Dataclass for storing the configuration required to create callbacks.

    Attributes:
        root_dir (Path): Base directory for callback-related artifacts.
        tensorboard_root_log_dir (Path): Directory where TensorBoard logs will be saved.
        checkpoint_model_filepath (Path): Full filepath where the model checkpoint will be stored.
    """
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    factor: float = 0.1
    patience: int = 5
    min_lr: float = 1e-7

    patience_early_stopping: int = 10
    restore_best_weights: bool = True
    verbose_early_stopping: int = 1
    checkpoint_monitor: str = 'val_loss'
    checkpoint_save_best_only: bool = True
    checkpoint_verbose: int = 1


@dataclass(frozen=True)
class Training_Config:
    """
    Dataclass for storing the configuration required to train the model.

    Attributes:
        root_dir (Path): Base directory for training artifacts.
        trained_model_path (Path): Path where the trained model will be saved.
        update_base_model (Path): Path to the updated base model file.
        training_data (Path): Path to the training data.
        param_image_size (list): List of image sizes for training.
        param_batch_size (int): Batch size for training.
        param_epochs (int): Number of epochs for training.
    """ 
    root_dir: Path
    trained_model_path: Path
    update_base_model: Path 
    training_data: Path
    param_image_size: list
    param_batch_size: int
    param_epochs: int 
    params_augmentation: bool 
    param_learning_rate: float


@dataclass
class Model_Evaluation_Config:
    """
    Configuration dataclass for model evaluation stage.

    Attributes:
        root_dir (Path): Root directory for storing evaluation artifacts.
        report_file_path (Path): Path to save the main evaluation report YAML/JSON.
        threshold_accuracy (float): Minimum accuracy threshold to consider model acceptable.
        scores_file_dir (Path): Directory to store evaluation scores.
        scores_file (str): Filename for the evaluation scores JSON.
        report_file_dir (Path): Directory to store detailed evaluation reports.
        report_file (str): Filename for the evaluation report JSON.
        mlflow_tracking_uri (str): MLflow tracking server URI for logging metrics and models.
        mlflow_experiment_name (str): Name of the MLflow experiment.
        all_params (Dict): Dictionary of all hyperparameters and config values for reference/logging.
        param_image_size (List[int]): Image size used during training/evaluation (Height, Width, Channels).
        param_batch_size (int): Batch size used during evaluation.
        training_data_path (Path): Path to the validation dataset for model evaluation.
    """
    root_dir: Path
    report_file_path: Path
    report_file_dir: Path
    report_file: str
    training_data: Path

    scores_file_dir: Path
    scores_file: str

    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    all_params: dict
    param_image_size: list
    param_batch_size: int
    threshold_accuracy: float 
     
