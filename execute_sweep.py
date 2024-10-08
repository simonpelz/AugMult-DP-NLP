import os
import wandb
from argparse import ArgumentParser

from init_sweep import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("sweep_id", help="pass the previously initialized wandb sweep_id to start sweep")
    args = parser.parse_args()

    os.environ["WANDB_DIR"] = os.path.abspath("")
    os.environ["MODEL_DIR"] = './/'

    os.environ["MAX_PHYS_BATCHSIZE"] = "4096"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    wandb.login()
    wandb.agent(args.sweep_id, function=main, count=1)