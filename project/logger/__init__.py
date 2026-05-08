import logging
import os
from datetime import datetime

# Create log folder and file
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_path, exist_ok=True)

log_file_path = os.path.join(log_path, log_file)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
    level=logging.INFO
)

# log message
logging.info("Logging Message Successful")


## [10/28/2025 12:07:52] [INFO] [app.py:18] - Logging Message Successful