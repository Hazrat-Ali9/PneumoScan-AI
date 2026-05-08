import sys
from aiohttp_retry import Union
import tensorflow as tf
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
from project.exception import CustomException
from project.entity.config import Training_Config
from project.utils import create_directories, tf_dataset
from project.logger import logging
from project.components.prepare_basemodel import total_loss, dice_coef
from project.configeration import Configeration_Manager
from project.components.callbacks import Call_Backs



class Training:
    """
    Handles the model training lifecycle, including data preparation, 
    model compilation, and execution of the training loop.
    """
    def __init__(self, config: Training_Config):
        """
        Initializes the Training class with configuration parameters.

        Args:
            config (Training_Config): Configuration object containing paths and hyperparameters.
        """
        self.config = config
        self.model = None

    def get_base_model(self):
        """
        Loads the pre-defined base model from disk and compiles it with 
        specified optimizer, loss function, and metrics.

        Raises:
            CustomException: If the model file is missing or compilation fails.
        """
        try:
            logging.info("Loading and compiling the base model...")
            self.model = tf.keras.models.load_model(str(self.config.update_base_model), compile=False)
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.param_learning_rate),
                loss=total_loss,
                metrics=[dice_coef, 'accuracy']
            )
        except Exception as e:
            raise CustomException(e, sys)

    def train_valid_generator(self):
        """
        Discovers image and mask files, performs a train-test split, 
        and wraps them into high-performance tf.data.Dataset objects.

        Returns:
            tuple: A tuple containing (train_dataset, val_dataset) as tf.data.Dataset instances.

        Raises:
            CustomException: If no images are found or data splitting fails.
        """
        try:
            training_path = Path(self.config.training_data)
            img_pattern = str(training_path / "images" / "*.png")
            mask_pattern = str(training_path / "masks" / "*.png")
            
            images = sorted(glob(img_pattern))
            masks = sorted(glob(mask_pattern))

            if len(images) == 0:
                raise Exception(f"No images found in {img_pattern}")

            train_x, val_x, train_y, val_y = train_test_split(
                images, masks, test_size=0.2, random_state=42
            )

            train_dataset = tf_dataset(train_x, train_y, batch_size=self.config.param_batch_size, training=True)
            val_dataset = tf_dataset(val_x, val_y, batch_size=self.config.param_batch_size, training=False)

            return train_dataset, val_dataset
        
        except Exception as e:
            raise CustomException(e, sys)

    from pathlib import Path
    from typing import Union

    @staticmethod
    def save_model_path(path: Union[str, Path], model: tf.keras.Model):
        path = Path(path) 
        
        # Now .parent will work perfectly
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model to the given path."""
        model.save(str(path))

    def train(self,train_data, val_data, callbacks: list = None):
        """
        Orchestrates the full training process: loads data, triggers the fit method, 
        and saves the final model outputs.

        Args:
            callbacks (list, optional): List of Keras callbacks (e.g., EarlyStopping, Checkpoints).

        Returns:
            tf.keras.callbacks.History: The history object containing loss and metric values per epoch.

        Raises:
            CustomException: If any error occurs during the training loop.
        """
        try:
            if self.model is None:
                self.get_base_model()

            train_data, val_data = self.train_valid_generator()

            logging.info("Starting model training...")
            history = self.model.fit(
                train_data,
                epochs=self.config.param_epochs,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )

            create_directories(["final_model"])
            final_model_dir = Path("final_model")
            self.save_model(path=final_model_dir / "model.keras", model=self.model)

            logging.info("Training completed successfully.")
            return history
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    try:    
        config = Configeration_Manager()

        # Prepare callbacks (optional, uncomment if needed)
        callbacks_config = config.get_prepare_callback_config()
        callback_list = Call_Backs(config=callbacks_config).get_callbacks()

        # Training setup
        training_config = config.get_training_config()
        trainer = Training(config=training_config)

        # Prepare model and data
        trainer.get_base_model()
        train_data, val_data = trainer.train_valid_generator()
        trainer.train(train_data=train_data,val_data=val_data,callbacks=callback_list)

    except Exception as e:
        raise CustomException(e, sys)