import os
import time

import numpy as np
import torch
import torch.optim as optim

from augmult.data import non_dp_tokenize_dataloader, get_dataset, has_labels
from train import compute_task_metric, eval, default_metric_name
from util.early_stopper import EarlyStopping
from util.different_finetune_modes import get_param_setter_from_str, model_and_tokenizer
from util.logging_util import _get_path, track_time
from util.reproducibility import set_seed

import wandb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')


def run(dataset_name, epochs, batch_size, lr, trainable_param_setter, num_labels,K=1,max_grad_norm=None,transform_list=None, noise_multiplier=None, dataset_size=None, save_model=None, target_epsilon=None, early_stop_patience=10, model_name = "bert-base-uncased", glue=True, max_phys_batchsize=1500,total_steps=None,
        seed=None, eval_test=True, multi_label=False, metric=None,
        hf_path=None, hf_config=None, text_columns=None, label_column="label", splits=None):

    # ---------------- Initialisation ------------------
    generator = set_seed(seed)

    # Model, Tokenizer
    model, tokenizer = model_and_tokenizer(model_name, num_labels, multi_label=multi_label)
    total_p, trainable_p = trainable_param_setter(model)
    finetune_percent = trainable_p / total_p * 100

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Dataloaders
    dataset = get_dataset(dataset_name, glue = glue, precomputed_augs=False, hf_path=hf_path, hf_config=hf_config,
                          text_columns=text_columns, label_column=label_column, splits=splits)
    modified_trainset = dataset['train'].select(range(dataset_size)) if dataset_size is not None else dataset['train']
    # NOTE: the thesis baseline used shuffle=False (fixed order every epoch); kept for comparability
    train_loader = non_dp_tokenize_dataloader(modified_trainset, tokenizer, batch_size, generator=generator)
    valid_loader = non_dp_tokenize_dataloader(dataset['validation'], tokenizer, max_phys_batchsize)

    steps_per_epoch = len(train_loader)
    if epochs is None:
        if total_steps is None: raise ValueError("Specify either epochs or total_steps")
        epochs = -(total_steps// -steps_per_epoch)
        early_stop_patience = max(10,(epochs//4))

    # Logging
    n_samples = len(train_loader.dataset)
    sampling_rate = batch_size / n_samples

    wandb.config.update({"early_stop_patience":early_stop_patience,"epochs":epochs,'K': K,'batches per epoch': steps_per_epoch,
        "transform_list": "non-DP","finetune_percent":finetune_percent,"sampling_rate":sampling_rate,
        "seed": seed, "metric": metric or default_metric_name(dataset_name, multi_label)}, allow_val_change=True)

    early_stop = EarlyStopping(patience=early_stop_patience)

    # -------------- Training ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    steps = 0
    train_inputs = {
        'model': model,
        'task_name':dataset_name,
        'train_loader': train_loader,
        'optimizer': optimizer,
        'device': device,
        'steps': steps,
        'multi_label': multi_label,
        'metric': metric,
    }
    eval_kwargs = {'device': device, 'task_name': dataset_name, 'multi_label': multi_label, 'metric': metric}

    valid_metric = 0
    for i in range(epochs):
        # Train for 1 epoch
        train_inputs['steps'] = normal_train(**train_inputs)
        # Evaluate
        valid_metric, valid_loss = eval(model, valid_loader, **eval_kwargs)
        wandb.log({"epoch": i+1,"valid_metric": valid_metric,"valid_loss": valid_loss})
        # early stopping
        if early_stop.step(valid_loss):
            wandb.config.update({"Stopped early": True}, allow_val_change=True)
            break
        # save ckpt
        if save_model:
            d = wandb.config.as_dict()
            d["epoch"]=i+1
            d["module_state_dict"] = model.state_dict()
            torch.save(d,_get_path(filename=f"non-dp-{dataset_name}-{wandb.run.id}"))


    # ------------ Evaluation ---------------------

    wandb.log({"epsilon": float('inf')})

    test_split = dataset.get('test')
    if eval_test and test_split is not None and has_labels(test_split):
        test_loader = non_dp_tokenize_dataloader(test_split, tokenizer, max_phys_batchsize)
        test_metric, test_loss = eval(model, test_loader, **eval_kwargs)
        wandb.log({"test_metric": test_metric, "test_loss": test_loss})
    elif eval_test:
        wandb.config.update({"test_skipped": "no labelled test split"}, allow_val_change=True)

    if valid_metric: return valid_metric # returning for sweep setup


def debug(params):

    wandb.init(project=("non-dp-"+params['dataset_name']), name="non-dp")
    wandb.config.update(params)

    setter = "classifier_and_pooler" if params['model_name']== "bert-base-uncased" else "classifier_only"
    trainable_param_setter = get_param_setter_from_str(setter)

    params = {**params, "trainable_param_setter": trainable_param_setter}

    run(**params)
    wandb.finish()

#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------------    Non-DP train     ------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


def normal_train(
    model,
    train_loader,
    optimizer,
    device,
    steps,
    task_name,
    multi_label=False,
    metric=None,
):


    logging_interval = 10
    to_mean_metrics = []
    to_mean_losses = []
    model.train()

    # Time Logging
    timers = {name: {"this_batch": [], "log": []} for name in ("data_loading", "forward", "backward", "optimizer")}

    step = steps
    start_time = track_time()
    for batch in train_loader:

        start_time = track_time(timers["data_loading"],start_time)

        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        if multi_label: batch['labels'] = batch['labels'].float()
        optimizer.zero_grad(set_to_none=True)

        outputs = model(**batch)

        loss = outputs.loss

        batch_metric = compute_task_metric(outputs,batch,task_name,multi_label=multi_label,metric=metric)

        start_time = track_time(timers["forward"],start_time)

        # Backward pass
        loss.backward()
        start_time = track_time(timers["backward"],start_time)

        # Optimization
        optimizer.step()
        start_time = track_time(timers["optimizer"],start_time)

        # Logging loss and acc
        step += 1

        for times in timers.values():
            times["log"].append(sum(times["this_batch"]))
            times["this_batch"] = []

        to_mean_metrics.append(batch_metric)
        to_mean_losses.append(loss.item())
        if step % logging_interval == logging_interval-1:
            l, m = np.mean(to_mean_losses), np.mean(to_mean_metrics)
            stats = {
                "train_loss": l,
                "train_metric": m,
                "optimizer_time": np.mean(timers["optimizer"]["log"]),
                "forward_time": np.mean(timers["forward"]["log"]),
                "backward_time": np.mean(timers["backward"]["log"]),
                "data_loading_time": np.mean(timers["data_loading"]["log"]),
            }
            wandb.log(stats, step=step)

            # Reset the timers and accumulators (in place)
            for times in timers.values():
                times["this_batch"].clear()
                times["log"].clear()
            to_mean_metrics, to_mean_losses = [], []

    return step


def sweep(b_sizes,lrs,params):
    for batch in b_sizes:
        for lr in lrs:
            params["lr"] = lr
            params["batch_size"] = batch
            try:
                debug(params)
            except KeyboardInterrupt:
                # wandb sends KeyboardInterrupt when stopping run
                wandb.finish()
                time.sleep(5) # allow for double interrupt to exit
                pass

if __name__ == "__main__":
    # Requires CKPT_PATH (when save_model is set) and DATASET_DIR (for local datasets) in the environment.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    sst2 = {
        'dataset_name': "sst2",
        'glue': True,
        'num_labels': 2,
        }
    mednli = {
        'dataset_name': "mednli",
        'glue': False,
        'num_labels': 3,
        }
    common = {
        **sst2,
        'max_phys_batchsize': 4096, # 5500 still not out of memory!
        'model_name': "bert-base-uncased",
        'seed': 0,
    }
    minimal = {
        **common,
        'epochs': 3,
        'dataset_size': 3000,
        'early_stop_patience': 100,

        'lr': 0.001,
        'batch_size': 100,
        }

    params = {
        **common,
        #'epochs': 50,
        'epochs': None,
        'total_steps': 2000,
        'dataset_size': None,
        #'early_stop_patience': 10,
        'save_model': "non-dp",

        #'lr': 0.001,
        #'batch_size': 16,
        }

    # sweep sst2
    batch_sweep = [32]
    lr_sweep = [0.001,0.1]
    sweep(batch_sweep, lr_sweep,params)

 #--------------------------------------------------
    # test roberta
    common = {
        **mednli,
        'max_phys_batchsize': 2048, # 5500 still not out of memory! (bert base)
        'model_name': "roberta-large",
        'save_model': None,
        'seed': 0,
    }
    params = {
        **common,
        #'epochs': 50,
        'epochs': None,
        'total_steps': 100000,
        'dataset_size': None,
        #'early_stop_patience': 10,
        'save_model': "non-dp-roberta",

        'lr': 0.001,
        'batch_size': 2,
        }
    sweep([4],[0.001,0.00001],params)
