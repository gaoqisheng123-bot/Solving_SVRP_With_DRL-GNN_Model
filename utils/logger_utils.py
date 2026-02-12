import logging
import os

def setup_logger(save_dir, log_filename='vrp_log.txt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, log_filename)
    
    # Avoid adding handlers if logger already has them (e.g., in Jupyter)
    logger = logging.getLogger()
    if not logger.handlers: # Setup handlers only if not already configured
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler() 
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger