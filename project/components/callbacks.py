import os
import sys
import time
import tensorflow as tf
from project.entity.config import Prepare_Callback_Config
from project.exception import CustomException
from project.constants import *
from project.configeration import Configeration_Manager
from project.utils import create_directories


class Call_Backs:
    """
    Manages the creation of Keras callbacks for model training.

    This class provides properties to generate individual callbacks and a 
    method to retrieve a compiled list of all callbacks for use in model.fit().

    Args:
        config (PrepareCall_backConfig): Configuration object containing paths 
                                         and parameters for callbacks.
    """

    def __init__(self, config: Prepare_Callback_Config):
        """Initializes the Call_Backs manager with the provided configuration."""
        try:
            self.config = config
        except Exception as e:
            raise CustomException(e, sys)

    @property
    def create_TensorBoard_callback(self) -> tf.keras.callbacks.TensorBoard:
        """
        Creates a TensorBoard callback with a timestamped log directory.

        Returns:
            tf.keras.callbacks.TensorBoard: Callback for visualizing training logs.
        """
        try:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            # Creating a sub-directory with a timestamp to avoid overwriting previous runs
            tb_log_dir = os.path.join(
                self.config.tensorboard_root_log_dir,
                f"tb_logs_at_{timestamp}"
            )
            return tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir)
        except Exception as e:
            raise CustomException(e, sys)

    @property
    def create_ModelCheckpoint_callback(self) -> tf.keras.callbacks.ModelCheckpoint:
        """
        Creates a ModelCheckpoint callback to save the best model weights.

        Returns:
            tf.keras.callbacks.ModelCheckpoint: Callback to save the model file.
        """
        try:    
            return tf.keras.callbacks.ModelCheckpoint(
                filepath=Path(self.config.checkpoint_model_filepath),
                save_best_only=bool(self.config.checkpoint_save_best_only),
                monitor=str(self.config.checkpoint_monitor),
                verbose=int(self.config.checkpoint_verbose)
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    @property
    def create_ReduceLROnPlateau_callback(self) -> tf.keras.callbacks.ReduceLROnPlateau:
        """
        Creates a ReduceLROnPlateau callback to dynamically adjust learning rate.

        Behavior:
            Reduces the learning rate by a factor of 0.1 if validation loss 
            does not improve for 5 consecutive epochs.

        Returns:
            tf.keras.callbacks.ReduceLROnPlateau: Callback for LR scheduling.
        """
        try:    
            return tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.config.checkpoint_monitor,
                factor=float(self.config.factor),
                patience=int(self.config.patience),
                min_lr=float(self.config.min_lr),
                verbose=int(self.config.checkpoint_verbose)
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    @property
    def create_EarlyStopping_callback(self) -> tf.keras.callbacks.EarlyStopping:
        """
        Creates an EarlyStopping callback to prevent overfitting.

        Behavior:
            Stops training if validation loss does not improve for 10 
            consecutive epochs and restores the best found weights.

        Returns:
            tf.keras.callbacks.EarlyStopping: Callback to halt training early.
        """
        try:    
            return tf.keras.callbacks.EarlyStopping(
                monitor=str(self.config.checkpoint_monitor),
                patience=int(self.config.patience_early_stopping),
                restore_best_weights=bool(self.config.restore_best_weights),
                verbose=int(self.config.verbose_early_stopping)
            )
        except Exception as e:
            raise CustomException(e, sys)


    def get_callbacks(self) -> list:
        """
        Compiles all initialized callbacks into a single list.

        Returns:
            list: A list containing TensorBoard, ModelCheckpoint, 
                  ReduceLROnPlateau, and EarlyStopping callbacks.
        """
        try:    
            call_back_list = [
                self.create_TensorBoard_callback, 
                self.create_ModelCheckpoint_callback,
                self.create_ReduceLROnPlateau_callback,
                self.create_EarlyStopping_callback
            ]
            return call_back_list
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        config_manager = Configeration_Manager()
        config = config_manager.get_prepare_callback_config()
        callbacks_manager = Call_Backs(config)
        callbacks_list = callbacks_manager.get_callbacks()
        print("Callbacks created successfully:")
        for callback in callbacks_list:
            print(f"- {callback.__class__.__name__}")
    except Exception as e:
        print(f"Error creating callbacks: {e}")