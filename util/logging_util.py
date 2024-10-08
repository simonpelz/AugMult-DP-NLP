import logging
import os
import time

import wandb

CKPT_PATH = os.environ.get("CKPT_PATH","./logs_and_ckpts/ckpts")

def _get_path(filename):
    return f"{CKPT_PATH}/{filename}.ckpt"

def save_ckpt(ckpt_dict, privacy_engine, model, filename,):
    privacy_engine.save_checkpoint(module=model,checkpoint_dict=ckpt_dict, path = _get_path(filename))


def remove_ckpt(filename):
    path = _get_path(filename)
    if os.path.exists(path):
        os.remove(path)



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



def track_time(time_dict=None,start_time=None):
    if time_dict is not None:
        if start_time is None: pass
        if "this_batch" not in time_dict:
            time_dict["this_batch"] = []
        delta = time.time() - start_time
        time_dict["this_batch"].append(delta)
    return time.time()


def _name(aug):
    if not hasattr(aug,"__self__"):
        return "unaugmented"
    else:
        return aug.__self__.name
    
    
def aug_name(aug_or_list):
    if type(aug_or_list) is list:
        ret = []
        for aug in aug_or_list:
            ret.append(_name(aug))
    else:
        ret = _name(aug_or_list)
    return ret

