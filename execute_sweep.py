import os
import wandb
from argparse import ArgumentParser

from init_sweep import main

def check_dirs():
    dirs = [os.getenv("WANDB_DIR"),os.getenv("MODEL_DIR"),os.getenv("CKPT_PATH")]
    for p in dirs:
        if not os.path.isdir(p):
            return False
    return True

if __name__ == "__main__":
    # Args
    parser = ArgumentParser()
    parser.add_argument("sweep_id", help="pass the previously initialized wandb sweep_id to start sweep")
    args = parser.parse_args()

    # Path handling
    os.environ["WANDB_DIR"] = "./logs_and_ckpts"
    os.environ["MODEL_DIR"] = './augmentation_models/'
    os.environ["CKPT_PATH"] = "./logs_and_ckpts/ckpts"
    if not check_dirs(): raise EnvironmentError

    # Physical env
    os.environ["MAX_PHYS_BATCHSIZE"] = "4096" 
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Requeue behaviour
    if os.environ.get("SLURM_JOB_ID") is None: raise EnvironmentError()
    print(os.environ.get('SLURM_RESTART_COUNT', '0')=='0')
    if os.environ.get('SLURM_RESTART_COUNT', '0') == '0':
        os.environ["CKPT_ID"] = os.environ.get("SLURM_JOB_ID")

    wandb.login()
    wandb.agent(args.sweep_id, function=main, count=1)