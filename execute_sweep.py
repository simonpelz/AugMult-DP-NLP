import os
import wandb
import signal
from argparse import ArgumentParser

from init_sweep import main

def check_dirs():
    dirs = [os.getenv("WANDB_DIR"),os.getenv("MODEL_DIR"),os.getenv("CKPT_PATH")]
    for p in dirs:
        if not os.path.isdir(p):
            return False
    return True

def signal_handler(signum, frame):
    wandb.mark_preempting()
    exit(1)


if __name__ == "__main__":
    # Args
    parser = ArgumentParser()
    parser.add_argument("sweep_id", help="pass the previously initialized wandb sweep_id to start sweep")
    args = parser.parse_args()

    # Path handling
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None: raise EnvironmentError()

    os.environ["WANDB_DIR"] = "REPLACE"
    os.environ["MODEL_DIR"] = 'REPLACE'
    ckpt_path = f"REPLACE/{job_id}/ckpts"
    os.makedirs(ckpt_path, exist_ok = True)
    os.environ["CKPT_PATH"] = ckpt_path
    if not check_dirs(): raise EnvironmentError

    # Physical env
    os.environ["MAX_PHYS_BATCHSIZE"] = "REPLACE" 
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Requeue behaviour
    signal.signal(signal.SIGTERM, signal_handler)
    if os.environ.get('SLURM_RESTART_COUNT', '0') != '0':
        ckpt_files = os.listdir(ckpt_path)
        n = len(ckpt_files)
        if n == 1:
            os.environ["CKPT_ID"] = ckpt_files[0].split('.')[0]
            print("Found ckpt. Resuming training...")
        elif n == 0: print("Job restarted, but no ckpt found. Starting from scratch...")
        elif n > 1: 
            print(f"Ambiguous ckpts. found {n} files but expected 1. Exiting...")
            exit(1)
    

    wandb.login()
    wandb.agent(args.sweep_id, function=main, count=1)