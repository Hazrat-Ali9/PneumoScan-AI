from project.logger import logging 
import sys 

class CustomException(Exception):
    """
    Custom Exception class for detailed error tracking.

    This class captures:
    - The error message
    - The exact line number where the error occurred
    - The file name where the error originated

    Usage:
        try:
            # Some risky code
        except Exception as e:
            raise CustomException(e, sys)
    """
    def __init__(self, error_message, error_details: sys):
        self.error_message = str(error_message)
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename  
        logging.error(self.__str__())  

    def __str__(self):
        return f"[{self.file_name}] - [Line {self.lineno}] - Error: {self.error_message}"
