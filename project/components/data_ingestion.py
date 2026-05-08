import os
import sys
import zipfile
import gdown
from project.logger import logging
from project.exception import CustomException
from project.entity.config import Data_Ingestion_Config
from project.configeration import Configeration_Manager


class Data_Ingestion:
    """
    Handles the data ingestion pipeline: downloading dataset from Google Drive 
    and extracting the downloaded zip file.

    Attributes:
        config (DataIngestionConfig): Configuration object containing paths and URLs.
    """

    def __init__(self, config: Data_Ingestion_Config):
        """
        Initialize the DataIngestion class with configuration settings.

        Args:
            config (DataIngestionConfig): Configuration parameters for data ingestion.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            logging.info("Initializing Data Ingestion")
            self.config = config
        except Exception as e:
            raise CustomException(e, sys)

    def download_data(self):
        """
        Downloads dataset from a Google Drive shareable link using gdown.

        Steps:
            1. Create root directory if not exists.
            2. Extract file ID from Google Drive URL.
            3. Construct direct download URL.
            4. Download the zip file to local_data_file.

        Raises:
            CustomException: If the download process fails.
        """
        try:
            data_url = self.config.source_url
            zip_down_dir = self.config.local_data_file

            os.makedirs(self.config.root_dir, exist_ok=True)
            logging.info(f"Downloading data from {data_url} to {zip_down_dir}")

            file_id = data_url.split('/')[-2]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            gdown.download(download_url, zip_down_dir)
            logging.info("Download completed successfully!")
        except Exception as e:
            raise CustomException(e, sys)

    def extract_zip_file(self):
        """
        Extracts the downloaded zip file into the specified directory.

        Steps:
            1. Create unzip directory if not exists.
            2. Extract all contents of the zip file into unzip_dir.

        Raises:
            CustomException: If extraction fails.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, 'r') as f:
                f.extractall(unzip_path)

            logging.info(f"Extraction completed at {unzip_path}")
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    try:
        config = Configeration_Manager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Data_Ingestion(data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()
    except Exception as e:
        raise CustomException(e, sys)