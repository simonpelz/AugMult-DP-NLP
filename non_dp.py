import os
import time

import torch
import torch.optim as optim

from augmult.data import non_dp_tokenize_dataloader, get_dataset
from util.early_stopper import EarlyStopping
from util.different_finetune_modes import get_param_setter_from_str, model_and_tokenizer
from util.logging_util import _get_path

import wandb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')


def run(dataset_name, epochs, batch_size, lr, trainable_param_setter, num_labels,K=1,max_grad_norm=None,transform_list=None, noise_multiplier=None, dataset_size=None, save_model=None, target_epsilon=None, early_stop_patience=10, model_name = "bert-base-uncased", glue=True, max_phys_batchsize=1500,total_steps=None):    

    # ---------------- Initialisation ------------------
    # Model, Tokenizer
    model, tokenizer = model_and_tokenizer(model_name, num_labels)
    total_p, trainable_p = trainable_param_setter(model)
    finetune_percent = trainable_p / total_p * 100

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Dataloaders
    precomputed = False
    dataset = get_dataset(dataset_name, glue = glue, precomputed_augs=precomputed)
    modified_trainset = dataset['train'].select(range(dataset_size)) if dataset_size is not None else dataset['train']
    train_loader = non_dp_tokenize_dataloader(modified_trainset, tokenizer, batch_size,)
    valid_loader = non_dp_tokenize_dataloader(dataset['validation'], tokenizer, max_phys_batchsize)

    steps_per_epoch = len(train_loader)
    if epochs is None:
        if total_steps is None: raise ValueError
        epochs = -(total_steps// -steps_per_epoch)
        early_stop_patience = max(10,(epochs//4))

    # Logging
    n_samples = len(train_loader.dataset)
    sampling_rate = batch_size / n_samples

    wandb.config.update({"early_stop_patience":early_stop_patience,"epochs":epochs,'K': K,'batches per epoch': steps_per_epoch,
        "transform_list": "non-DP","finetune_percent":finetune_percent,"sampling_rate":sampling_rate,}, allow_val_change=True)
    
    # ------------- Remnants of copy Make DP with AugMult -----------------

    delta = 1 / n_samples
    early_stop = EarlyStopping(patience=early_stop_patience)

    resume_ckpt = None #os.environ.get("CKPT_ID",None)
    ckpt_dict = {}


    # -------------- Training ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    start_epoch=ckpt_dict.get("epoch",0) # because logging starts counting at 1 not 0, this is correct
    steps = steps_per_epoch*start_epoch
    train_inputs = {
        'model': model,
        'task_name':dataset_name,
        'train_loader': train_loader,
        'optimizer': optimizer,
        'device': device,
        'K': K,
        'max_phys_batch_size': max_phys_batchsize,
        'steps': steps, # possibly overwrite some logged steps when requeueing (real batchsize unknown)
    }

    valid_metric = 0
    for i in range(start_epoch,epochs):
        # Train for 1 epoch
        train_inputs['steps'] = normal_train(**train_inputs)
        # Evaluate
        valid_metric, valid_loss = eval(model, valid_loader,task_name=dataset_name, device=device)
        wandb.log({"epoch": i+1,"valid_metric": valid_metric,"valid_loss": valid_loss})
        # early stopping
        if early_stop.step(valid_loss): 
            wandb.config.update({"Stopped early": True})
            break
        # save ckpt
        if save_model:
            d = wandb.config.as_dict()
            d["epoch"]=i+1
            d["module_state_dict"] = model.state_dict()
            torch.save(d,_get_path(filename=f"non-dp-{dataset_name}-{wandb.run.id}"))


    # ------------ Evaluation ---------------------

    real_eps = float('inf')
    wandb.log({"epsilon": real_eps, "delta":delta})

    if valid_metric: return valid_metric # returning for sweep setup


def debug(params):

    wandb.init(project=("non-dp-"+params['dataset_name']), name="non-dp")
    wandb.config.update(params)   

    setter = "classifier_and_pooler" if params['model_name']== "bert-base-uncased" else "classifier_only"
    trainable_param_setter = get_param_setter_from_str(setter)

    params.update({"trainable_param_setter":trainable_param_setter})

    run(**params)
    wandb.finish()

#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------------    Non-DP train     ------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

import torch
import numpy as np
from train import compute_task_metric

import wandb
from util.logging_util import track_time


def normal_train(
    model,
    train_loader,
    optimizer,
    device,
    K,
    max_phys_batch_size,
    steps,
    task_name,
):
    

    logging_interval = 10
    to_mean_metrics = []
    to_mean_losses = []
    model.train()
    
    # Time Logging
    data_loading_times, forward_times, backward_times, optimizer_times = {"this_batch":[],"log":[]},{"this_batch":[],"log":[]},{"this_batch":[],"log":[]},{"this_batch":[],"log":[]},

    step = steps
    start_time = track_time()
    for batch in train_loader:
    
        start_time = track_time(data_loading_times,start_time)

        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True) 

        outputs = model(**batch)

        loss = outputs.loss 

        metric = compute_task_metric(outputs,batch,task_name)

        start_time = track_time(forward_times,start_time)
        
        # Backward pass
        loss.backward()
        start_time = track_time(backward_times,start_time)

        # Optimization
        optimizer.step()
        start_time = track_time(optimizer_times,start_time)


        # Logging loss and acc
        is_updated = True
        if is_updated: 
            step += 1  

            for times in (data_loading_times,forward_times,backward_times,optimizer_times):
                times["log"].append(sum(times["this_batch"]))
                times["this_batch"] = []

            to_mean_metrics.append(metric)
            to_mean_losses.append(loss.item())
            if step % logging_interval == logging_interval-1:
                l, m = np.mean(to_mean_losses), np.mean(to_mean_metrics)
                stats = {
                    "train_loss": l,
                    "train_metric": m,
                    "optimizer_time": np.mean(optimizer_times["log"]),
                    "forward_time": np.mean(forward_times["log"]),
                    "backward_time": np.mean(backward_times["log"]),
                    "data_loading_time": np.mean(data_loading_times["log"]),
                }
                wandb.log(stats, step=step)
                
                # Reset the timers and accumulators
                for times in (data_loading_times,forward_times,backward_times,optimizer_times): times = {"this_batch":[],"log":[]}
                to_mean_metrics, to_mean_losses = [], []
      
    return step



def eval(model, eval_loader,device,task_name):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    losses = []
    test_metric = []

    with torch.no_grad():
        for batch in eval_loader:

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss 
            metric = compute_task_metric(outputs,batch,task_name)

            losses.append(loss.item())
            test_metric.append(metric)

    test_metric_avg = np.mean(test_metric)
    losses_avg  = np.mean(losses)

    model.train()

    return (test_metric_avg,losses_avg)
                
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
    os.environ['CKPT_PATH'] = "/home/spelz/AugMult_DP_NLP/logs_and_ckpts/ckpts"
    os.environ['DATASET_DIR'] = "datasets/processed"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


