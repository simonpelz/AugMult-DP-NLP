import os
import wandb
import signal
from argparse import ArgumentParser

from init_sweep import main

REQUIRED_DIRS = ("WANDB_DIR", "MODEL_DIR", "DATASET_DIR", "JOBS_DIR")


def check_dirs():
    missing = [v for v in REQUIRED_DIRS if not os.getenv(v) or not os.path.isdir(os.getenv(v))]
    if missing:
        raise EnvironmentError(f"Set these environment variables to existing directories: {missing}")

def signal_handler(signum, frame):
    wandb.mark_preempting()
    print("REQUEUEING")
    exit(1)


if __name__ == "__main__":
    # Args
    parser = ArgumentParser()
    parser.add_argument("sweep_id", help="pass the previously initialized wandb sweep_id to start sweep")
    args = parser.parse_args()

    # Path handling
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None: raise EnvironmentError("execute_sweep.py is meant to run inside a SLURM job (SLURM_JOB_ID unset)")

    # Paths come from the environment (previously hard-coded to the thesis cluster):
    # WANDB_DIR, MODEL_DIR, DATASET_DIR, and JOBS_DIR (per-job checkpoints go to $JOBS_DIR/<job_id>/ckpts)
    check_dirs()
    ckpt_path = os.path.join(os.environ["JOBS_DIR"], str(job_id), "ckpts")
    os.makedirs(ckpt_path, exist_ok = True)
    os.environ["CKPT_PATH"] = ckpt_path

    # Physical env
    os.environ.setdefault("MAX_PHYS_BATCHSIZE", "4096")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Requeue behaviour
    signal.signal(signal.SIGTERM, signal_handler)
    if os.environ.get('SLURM_RESTART_COUNT', '0') != '0':
        # FIXME this is broken
        ckpt_files = os.listdir(ckpt_path)
        n = len(ckpt_files)
        if n == 1:
            os.environ["CKPT_ID"] = ckpt_files[0].split('.')[0]
            print("Found ckpt. Resuming training...")
        elif n == 0: 
            print("Job restarted, but no ckpt found. Assuming timelimit ran out. currently not supported because of requeue behaviour")#Starting from scratch...")
            exit(0)
        elif n > 1: 
            print(f"Ambiguous ckpts. found {n} files but expected 1. Exiting...")
            exit(1)
    

    wandb.login()
    wandb.agent(args.sweep_id, function=main, count=1)