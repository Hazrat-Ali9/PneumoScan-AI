import os
import sys
import json
import yaml
import base64
import joblib

from pathlib import Path
from typing import Any, List, Union

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from project.logger import logging
from project.exception import CustomException

# ============================
# Y A M L   F U N C T I O N S
# ============================

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns ConfigBox."""
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            
        # If file is empty, yaml.safe_load returns None
        if content is None:
            logging.warning(f"YAML file {path_to_yaml} is empty. Returning empty ConfigBox.")
            content = {} # Initialize as empty dict so ConfigBox works
            
        logging.info(f"YAML file loaded successfully: {path_to_yaml}")
        return ConfigBox(content)

    except Exception as e:
        raise CustomException(e, sys)


# ====================================
# D I R E C T O R I E S   H A N D L I N G
# ====================================

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Creates a list of directories if they do not already exist.

    Args:
        path_to_directories (list): A list of directory paths.
        verbose (bool): If True, logs the creation of each directory. Defaults to True.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logging.info(f"Created directory at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


# =============================
# J S O N   F U N C T I O N S
# =============================

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a formatted JSON file.

    Args:
        path (Path): Path where the JSON file will be saved.
        data (dict): Dictionary data to be serialized.

    Raises:
        CustomException: If the path is invalid or data is not serializable.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its content as a ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Content of the JSON file.

    Raises:
        CustomException: If the file is missing or corrupted.
    """
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logging.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)


# ===============================
# B I N A R Y   F U N C T I O N S
# ===============================

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data in binary format using joblib.

    Args:
        data (Any): The object to be saved (e.g., a trained model).
        path (Path): Path where the binary file will be saved.

    Raises:
        CustomException: If the object cannot be pickled/saved.
    """
    try:
        joblib.dump(value=data, filename=path)
        logging.info(f"Binary file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads binary data (e.g., a model) from a file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: The deserialized Python object.

    Raises:
        CustomException: If the file cannot be read or is incompatible.
    """
    try:
        data = joblib.load(path)
        logging.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys)


# ==================
# U T I L I T Y
# ==================

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the file size in a human-readable KB format.

    Args:
        path (Path): Path to the file.

    Returns:
        str: File size string (e.g., "~ 24 KB").
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        raise CustomException(e, sys)


# =================================
# B A S E 6 4   E N C O D E / D E C O D E
# =================================

@ensure_annotations
def decodeImage(imgstring: str, fileName: str) -> None:
    """
    Decodes a Base64 encoded string and saves it as an image file.

    Args:
        imgstring (str): The Base64 encoded image string.
        fileName (str): Path/name of the file to save the decoded image.
    """
    try:
        imgdata = base64.b64decode(imgstring)
        with open(fileName, 'wb') as f:
            f.write(imgdata)
        logging.info(f"Image decoded and saved to: {fileName}")
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """
    Encodes an image file into a Base64 bytes object.

    Args:
        croppedImagePath (str): Path to the image file to be encoded.

    Returns:
        bytes: The Base64 encoded representation of the image.
    """
    try:
        with open(croppedImagePath, "rb") as f:
            return base64.b64encode(f.read())
    except Exception as e:
        raise CustomException(e, sys)


## ------------------------------Tensorflow-Choice---------------------------
import tensorflow as tf

def parse_data(img_path, mask_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [256, 256])
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

def sync_augment(img, mask):

    seed = tf.random.experimental.stateless_split(tf.random.uniform([2], maxval=10000, dtype=tf.int32), num=1)[0]

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=seed)

    img = tf.image.random_brightness(img, max_delta=0.2)

    return img, mask

def tf_dataset(x, y, batch_size=8, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    dataset = dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.map(sync_augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset