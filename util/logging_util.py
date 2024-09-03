import logging
import time

import wandb

def get_file_logger(logger_name, log_file, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(logger_name)
    
    # Set the logging level
    logger.setLevel(level)
    
    # Create a file handler that writes to the specified log file in append mode
    file_handler = logging.FileHandler(f"./logs_and_ckpts/logs/{log_file}", mode='a')
    
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


def log_metrics(step, metrics, logger=None):
    """
    Logs the metrics to both the console and wandb.
    
    Args:
    - step: The current step or epoch.
    - metrics: A dictionary of metrics to log.
    - logger: Optional logger to use for console logging.
    """
    
    # Log to WandB
    wandb.log(metrics, step=step)
        
    
    # Log to console if logger is provide
    log_str = f"Step {step}, " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    if logger:
        logger.info(log_str)
    else:
        print(log_str)



def track_time(time_list=None,start_time=None):
    if time_list is not None:
        if start_time is None:
            pass
        delta = time.time() - start_time
        time_list.append(delta)
    return time.time()



