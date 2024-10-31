import wandb
from augmult.augmentations import get_transforms_from_str
from run_experiment import wrap_run_cuda
from util.different_finetune_modes import get_param_setter_from_str

import os

PROJECT_NAME = "mednli"
DATASET = "mednli"
GLUE = False
NUM_LABELS = 3


def main():
    # wrapping needed to pass wandb.config, couldnt simplify
    # + logic to resume run when preempted
    run_id = os.environ.get("SLURM_JOB_ID")
    if run_id: wandb.init(resume='allow',id=run_id)
    else: wandb.init()
    objective(wandb.config)


def objective(config):
    max_phys_batchsize = int(os.environ.get('MAX_PHYS_BATCHSIZE', "4096"))
    trainable_param_setter = get_param_setter_from_str("classifier_and_pooler")
    transform_list = get_transforms_from_str(config.get("transform_name", None))
    K = len(transform_list)

    params = config.as_dict()
    params.update({"epochs": None, "K":K,"transform_list":transform_list,"trainable_param_setter":trainable_param_setter
                   ,"max_phys_batchsize":max_phys_batchsize})
    params.pop("transform_name", None)

    return wrap_run_cuda(params) # pay attention to kwarg unwrapping or not


sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "valid_metric"},
    "parameters": {

        "total_steps": {"value": 1500},

        "batch_size": {"value": 512},

        "lr": {"value": 133.33},
        # TODO comment out noise modifier
        "max_grad_norm": {"value":0.0075},
        "transform_name": {"value": None},

        "target_epsilon": {"value": 8},   

        "dataset_size": {"value": None},   
        "save_model": {"value": "sanitycheckimpl"},
        
        # Static
        "dataset_name": {"value": DATASET},   
        "num_labels": {"value": NUM_LABELS},   
        "model_name": {"value": "bert-base-uncased"},   
        "glue": {"value": GLUE},   
        #"logs_per_epoch": {"value": 10},   
        #"max_phys_batchsize": {"value": 4096}   
        #"noise_multiplier": {"value": None},   
    },
}

if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME,)
    print(f"{PROJECT_NAME}/{sweep_id}")