# Import the W&B Python Library and log into W&B
import gc
from uu import Error
import torch
import wandb
from augmult.augmentations import get_transforms_from_str
from run_experiment import run
from util.different_finetune_modes import get_param_setter_from_str

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

wandb.login()

PROJECT_NAME = "cola"
DATASET = "cola"
NUM_LABELS = 2

def wrap_run_cuda(params):
    try:
        score = run(**params)
    except Error as e:
        wandb.config.update({"failed_because":e})
        print("\n\n---------------\nError\n--------\n"+e)
        raise e
    
    torch.cuda.empty_cache()
    gc.collect()
    return score


# 1: Define objective/training function
def objective(config):

    trainable_param_setter = get_param_setter_from_str("classifier_and_pooler")
    transform_list = get_transforms_from_str(config.get("transform_name", None))
    lr = config.get("lr_times_norm",1) / config.get("max_grad_norm")
    K = len(transform_list)
    tmp_epochs = config.get("epochs",None)
    early_stop_patience = max(4,tmp_epochs//3) if tmp_epochs is not None else None
    config.update({"lr":lr,})

    params = config.as_dict()
    params.update({"epochs": None, "K":K,"transform_list":transform_list,"trainable_param_setter":trainable_param_setter
                   ,})
    params.pop("transform_name", None)
    params.pop("lr_times_norm", None)

    return wrap_run_cuda(params) # pay attention to kwarg unwrapping or not


def main():
    wandb.init(project=PROJECT_NAME,)
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "method": "bayes",  # Random search method
    "metric": {"goal": "maximize", "name": "valid_metric"},  # Corrected spelling of "maximize"
    "parameters": {

        "total_steps": {
            "distribution": "q_log_uniform_values",
            "min": 2000,
            "max": 10000
        },

        "batch_size": {
            "distribution": "q_log_uniform_values",
            "min": 32,
            "max": 4096
        },  # Logarithmic range for batch size

        "lr_times_norm": {
            "distribution": "log_uniform_values",
            "min": 0.3,
            "max": 3
        },  # Logarithmic range for learning rate times norm

        "target_epsilon": {"value": 8},  # Fixed parameter
        "max_grad_norm": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 5.0
        },  # Range for max grad norm

        "transform_name": {"values": ["cola",None]},

        "noise_multiplier": {"value": None},  # Fixed parameter
        "dataset_size": {"value": None},  # Fixed parameter
        "save_model": {"value": None},  # Fixed parameter
        
        # Static
        "dataset_name": {"value": DATASET},  # Fixed parameter
        "num_labels": {"value": NUM_LABELS},  # Fixed parameter
        "model_name": {"value": "bert-base-uncased"},  # Fixed parameter
        "glue": {"value": True},  # Fixed parameter
        "logs_per_epoch": {"value": 10},  # Fixed parameter
        "max_phys_batchsize": {"value": 2048}  # Fixed parameter
    },
}


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME,)

wandb.agent(sweep_id, function=main, count=10)