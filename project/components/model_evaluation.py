import sys
from mlflow.models import infer_signature
import numpy as np
import tensorflow as tf
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split

from project.configeration import Configeration_Manager
from project.exception import CustomException
from project.entity.config import Model_Evaluation_Config
from project.components.prepare_basemodel import dice_coef,dice_loss,focal_loss,total_loss
from project.utils import tf_dataset, save_json
from project.logger import logging
from project.entity.config import Model_Evaluation_Config
from project.utils import save_json


import mlflow
import dagshub
# dagshub.init(repo_owner='Ahmed2797', repo_name='PneumoScan-AI', mlflow=True)


class Evaluation:
    """
    Handles model evaluation, saving scores, and logging metrics/models to MLflow.
    """

    def __init__(self, config: Model_Evaluation_Config):
        self.config = config

    def _valid_generator(self):
        """Create validation dataset."""
        try:
            training_path = Path(self.config.training_data)
            img_pattern = str(training_path / "images" / "*.png")
            mask_pattern = str(training_path / "masks" / "*.png")
            
            images = sorted(glob(img_pattern))
            masks = sorted(glob(mask_pattern))

            if len(images) == 0:
                raise Exception(f"No images found in {img_pattern}")

            # Splitting to get validation set only
            _, val_x, _, val_y = train_test_split(
                images, masks, test_size=0.2, random_state=42
            )

            val_dataset = tf_dataset(val_x, val_y, batch_size=self.config.param_batch_size, training=False)
            return val_dataset
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def load_model(model_path: Path):
        """Load the trained model with custom objects."""
        return tf.keras.models.load_model(model_path, custom_objects={
            'total_loss': total_loss,
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss
        })

    def evalution(self):
        """
        Evaluate the model and save results in structured JSON format.
        """
        try:
            val_dataset = self._valid_generator()
            model_path = Path("artifacts/training/bestmodel.keras")
            model = self.load_model(model_path)

            logging.info(f"Starting evaluation on validation dataset using {model_path}...")
            results = model.evaluate(val_dataset, verbose=0)
            print("Evaluation results (Loss, Dice Coef, Accuracy):")
            print(results)
            
            # Index 0: Loss, Index 1: Dice_Coef, Index 2: Accuracy
            eval_metrics = {
                "loss": float(results[0]),
                "dice_score": float(results[1]),
                "accuracy": float(results[2])
            }
            
            logging.info(f"Results -> Loss: {eval_metrics['loss']:.4f}, "
                         f"Dice: {eval_metrics['dice_score']:.4f}, "
                         f"Acc: {eval_metrics['accuracy']:.4f}")

            # ৫. Save scores.json
            scores_path = Path(self.config.scores_file_dir) / self.config.scores_file
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(path=scores_path, data=eval_metrics)

            return eval_metrics

        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise CustomException(e, sys)


    def log_mlflow(self, results=None):
        """Log metrics and model to MLflow with a proper Signature."""
        if results is None:
            results = self.evalution()

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        with mlflow.start_run():
            # Log Params and Metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("val_loss", results["loss"])
            mlflow.log_metric("dice_score", results["dice_score"])
            mlflow.log_metric("val_accuracy", results["accuracy"])

            # Create Model Signature
            # We simulate a dummy input to let MLflow 'see' the shape
            input_example = np.random.rand(1, 256, 256, 3).astype(np.float32)
            model = self.load_model(Path("artifacts/training/bestmodel.keras"))
            prediction = model.predict(input_example)
            signature = infer_signature(input_example, prediction)

            # Log Model with Signature and New Naming Convention
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model_evaluation",
                signature=signature,
                registered_model_name="PneumoScan_V1"
            )
        
            logging.info("MLflow logging complete with Model Signature.")
    def run_evaluation_pipeline(self):
        """Orchestrate the entire evaluation process."""
        results = self.evalution()
        
        # Save JSON/YAML reports
        save_json(path=Path(self.config.scores_file_dir) / self.config.scores_file, data=results)
        
        # Log to MLflow
        if self.config.mlflow_tracking_uri:
            self.log_mlflow(results=results)
        else:
            logging.warning("MLflow tracking URI not provided. Skipping MLflow logging.")


if __name__ == "__main__":
    try:
        config_manager = Configeration_Manager()
        model_evaluation_config = config_manager.get_model_evaluation_config()

        evaluator = Evaluation(config=model_evaluation_config)
        evaluator.run_evaluation_pipeline()
    except Exception as e:
        raise CustomException(e, sys)
    
