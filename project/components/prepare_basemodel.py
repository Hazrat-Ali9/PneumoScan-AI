import os
import sys
import tensorflow as tf
from pathlib import Path
from typing import Tuple
from keras import layers, models, Input
from keras.applications import ResNet50

from project.configeration import Configeration_Manager
from project.entity.config import Prepare_Basemodel_Config
from project.exception import CustomException
from project.logger import logging

# ==========================================
# 1. SEGMENTATION METRICS & LOSSES
# ==========================================

from keras import ops
from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Calculates the Dice Coefficient for overlap measurement.
    
    The Dice Coefficient is a statistical tool which measures the similarity 
    between two sets of data (ground truth vs. prediction).
    
    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted mask (probabilities from sigmoid).
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        float: Dice coefficient value (Range [0, 1]).
    """
    y_true_f = ops.convert_to_tensor(y_true, dtype="float32")
    y_pred_f = ops.convert_to_tensor(y_pred, dtype="float32")
    
    y_true_f = ops.reshape(y_true_f, [-1])
    y_pred_f = ops.reshape(y_pred_f, [-1])
    
    intersection = ops.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (ops.sum(y_true_f) + ops.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Calculates Dice Loss as (1 - Dice Coefficient).
    
    Useful for segmentation tasks where class imbalance is high (small targets).
    
    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted mask.
        
    Returns:
        float: Dice loss value.
    """
    return 1.0 - dice_coef(y_true, y_pred)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Calculates Focal Loss for binary segmentation.
    
    Focal loss down-weights easy examples and focuses on hard-to-classify 
    pixels (like the edges of a lesion or pneumonia patch).
    
    Args:
        gamma: Focusing parameter. Higher values reduce loss for easy pixels.
        alpha: Balancing parameter to handle class imbalance.
        
    Returns:
        function: A callable loss function for Keras.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = ops.convert_to_tensor(y_true, dtype="float32")
        epsilon = K.epsilon()
        
        # Clip to avoid log(0)
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary Cross Entropy component
        # We include both positive and negative cases for binary segmentation
        bce = - (y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred))
        
        # Focal weighting: (1 - p)^gamma for positive pixels, (p)^gamma for negative
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        focal_weight = ops.power(1.0 - p_t, gamma)
        
        loss = alpha * focal_weight * bce
        
        # Average loss over all pixels in the image
        return ops.mean(loss)
        
    return focal_loss_fixed

def total_loss(y_true, y_pred):
    """
    Combined Hybrid Loss: Focal Loss + Dice Loss.
    
    This combination handles both pixel-level class imbalance (Focal) 
    and global structural overlap (Dice).
    
    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted mask.
        
    Returns:
        float: Combined loss value.
    """
    return focal_loss()(y_true, y_pred) + dice_loss(y_true, y_pred)



# ==========================================
# 2. RESNET50 U-NET PREPARATION
# ==========================================

class Prepare_Segmentation_Model:
    """
    Handles building a U-Net using ResNet50 as an encoder for segmentation.
    """
    def __init__(self, config: Prepare_Basemodel_Config):
        self.config = config

    def build_resnet50_unet(self) -> models.Model:
        """Constructs and compiles the ResNet50 U-Net architecture."""
        try:
            input_shape = self.config.param_image_size
            inputs = Input(shape=input_shape)

            # --- ENCODER (ResNet50) ---
            base_model = ResNet50(
                include_top=self.config.param_include_top, 
                weights=self.config.param_weight, 
                input_tensor=inputs
            )
            
            # Extract Skip Connection layers
            s1 = base_model.input[0]                                  # 256x256
            s2 = base_model.get_layer("conv1_relu").output           # 128x128
            s3 = base_model.get_layer("conv2_block3_out").output     # 64x64
            s4 = base_model.get_layer("conv3_block4_out").output     # 32x32
            
            # Bridge
            bridge = base_model.get_layer("conv4_block6_out").output # 16x16

            # --- DECODER PATH ---
            def decoder_block(input_tensor, skip_tensor, filters):
                x = layers.UpSampling2D((2, 2))(input_tensor)
                x = layers.Concatenate()([x, skip_tensor])
                x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
                return x

            d1 = decoder_block(bridge, s4, 512)  
            d2 = decoder_block(d1, s3, 256)      
            d3 = decoder_block(d2, s2, 128)      
            d4 = decoder_block(d3, s1, 64)       

            outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

            model = models.Model(inputs, outputs, name="ResNet50_U-Net")
            
            # --- COMPILATION ---
            # Using your hybrid loss and dice metric
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.param_learning_rate),
                loss=total_loss,
                metrics=[dice_coef, 'accuracy']
            )

            logging.info("ResNet50 U-Net model successfully built and compiled.")
            return model

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def save_model(path: Path, model: models.Model):
        """Persists the model to disk."""
        try:
            model.save(path)
            logging.info(f"Model saved to: {path}")
        except Exception as e:
            raise CustomException(e, sys)


# ==========================================
# 3. EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        config_manager = Configeration_Manager()
        prepare_base_model_config = config_manager.get_prepare_base_model_config()

        model_preparer = Prepare_Segmentation_Model(config=prepare_base_model_config)
        unet_model = model_preparer.build_resnet50_unet()
        
        # Save to the 'updated_base_model' path defined in config
        model_preparer.save_model(
            path=Path(prepare_base_model_config.update_base_model), 
            model=unet_model
        )
    except Exception as e:
        raise CustomException(e, sys)