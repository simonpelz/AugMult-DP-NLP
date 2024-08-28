import logging

def get_file_logger(logger_name, log_file, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(logger_name)
    
    # Set the logging level
    logger.setLevel(level)
    
    # Create a file handler that writes to the specified log file in append mode
    file_handler = logging.FileHandler(log_file, mode='a')
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)

    logger.info("---- New Experiment ------")
    
    return logger

def log_from_dict(logger, d):
    for k,v in d.items():
        logger.info(f"{k}: {v}")

