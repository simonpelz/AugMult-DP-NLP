import argparse
import gc
import os
import types
from uu import Error

import torch
import torch.optim as optim

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from augmult.augmentations import get_transforms_from_str
from train import train, eval
from augmult.data import dp_dataloader, init_mv_collate, non_dp_tokenize_dataloader, get_dataset
from util.early_stopper import EarlyStopping
from util.privacy_engine_util import _prepare_model_modified, empty_batch_handling, prepare_gradsamplers, add_noise
from util.logging_util import aug_name, load_ckpt, save_ckpt
from util.different_finetune_modes import get_param_setter_from_str, model_and_tokenizer

import wandb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')


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


def run(dataset_name, transform_list, K, epochs, batch_size, lr, max_grad_norm, trainable_param_setter, num_labels, noise_multiplier=None, dataset_size=None, save_model=None, target_epsilon=None, early_stop_patience=10, model_name = "bert-base-uncased", glue=True, max_phys_batchsize=1500,total_steps=None):    

    # ---------------- Initialisation ------------------
    # Model, Tokenizer
    model, tokenizer = model_and_tokenizer(model_name, num_labels)
    total_p, trainable_p = trainable_param_setter(model)
    finetune_percent = trainable_p / total_p * 100

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Dataloaders
    precomputed = None in transform_list
    dataset = get_dataset(dataset_name, glue = glue, precomputed_augs=precomputed)
    mv_train_loader = dp_dataloader(dataset["train"],dataset_size,tokenizer,transform_list,batch_size)
    valid_loader = non_dp_tokenize_dataloader(dataset['validation'], tokenizer, max_phys_batchsize)

    steps_per_epoch = len(mv_train_loader)
    if epochs is None:
        if total_steps is None: raise ValueError
        epochs = -(total_steps// -steps_per_epoch)
        early_stop_patience = 1000 #max(5,(epochs//2))

    # Logging
    n_samples = len(mv_train_loader.dataset)
    sampling_rate = batch_size / n_samples

    wandb.config.update({"early_stop_patience":early_stop_patience,"epochs":epochs,'K': K,'batches per epoch': steps_per_epoch,
        "transform_list": aug_name(transform_list),"finetune_percent":finetune_percent,"sampling_rate":sampling_rate,})
    
    # ------------- Make DP with AugMult -----------------

    delta = 1 / n_samples
    early_stop = EarlyStopping(patience=early_stop_patience)

    # Ensure DP compatibility
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()
   
    # Hack to load the custom AugmultGradSamplerModule
    privacy_engine._prepare_model = types.MethodType(_prepare_model_modified, privacy_engine)

    # Make Private
    make_private_params = {'module': model,'optimizer': optimizer,'data_loader': mv_train_loader,'max_grad_norm': max_grad_norm,'grad_sample_mode': "augmult",}
    if target_epsilon is None:
        dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(**make_private_params,noise_multiplier=noise_multiplier,)
    else:
        dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(**make_private_params,target_epsilon=target_epsilon,target_delta=delta,epochs=epochs)
    
    resume_ckpt = os.environ.get("CKPT_ID",None)
    ckpt_dict = load_ckpt(privacy_engine,dp_model,resume_ckpt) if resume_ckpt is not None else {}

    # Hack to modify Optimizer
    dp_optimizer.add_noise = types.MethodType(add_noise, dp_optimizer)

    wandb.config.update({"noise_multiplier": dp_optimizer.noise_multiplier})

    # Override the empty batch shapes provided by privacy engine
    mv_collate = init_mv_collate(tokenizer,transform_list,max_length=128,precomputed=precomputed)
    empty_batch_handling(mv_train_loader=mv_train_loader,dp_train_loader=dp_train_loader,mv_collate=mv_collate)

    # Custom Grad Samplers
    dp_model = prepare_gradsamplers(K,dp_model)

    # -------------- Training ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_model.to(device=device)

    start_epoch=ckpt_dict.get("epoch",0) # because logging starts counting at 1 not 0, this is correct
    steps = steps_per_epoch*start_epoch
    train_inputs = {
        'dp_model': dp_model,
        'task_name':dataset_name,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': K,
        'max_phys_batch_size': max_phys_batchsize,
        'steps': steps, # possibly overwrite some logged steps when requeueing (real batchsize unknown)
    }

    valid_metric = 0
    for i in range(start_epoch,epochs):
        # Train for 1 epoch
        train_inputs['steps'] = train(**train_inputs)
        # Evaluate
        valid_metric, valid_loss = eval(dp_model, valid_loader,task_name=dataset_name, device=device)
        wandb.log({"epoch": i+1,"valid_metric": valid_metric,"valid_loss": valid_loss})
        # early stopping
        if early_stop.step(valid_loss): 
            wandb.config.update({"Stopped early": True})
            break
        # save ckpt
        if save_model:
            d = wandb.config.as_dict()
            d["epoch"]=i+1
            save_ckpt(ckpt_dict=d,privacy_engine=privacy_engine,model=dp_model,filename=f"{wandb.run.id}")#f"{save_model}_{wandb.run.name}")

    # ------------ Evaluation ---------------------

    # Privacy
    try:
        real_eps = privacy_engine.accountant.get_epsilon(delta=delta)
    except OverflowError as error:
        real_eps = float('inf')
    wandb.log({"epsilon": real_eps, "delta":delta})

    if valid_metric: return valid_metric # returning for sweep setup


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation Multiplicity for DP-NLP Models")
    
    # Training Args
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in dataset')
    parser.add_argument('--dataset_size', type=int, default=None, help='Dataset size')

    # DP
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--noise_multiplier', type=float, default=None, help='Noise multiplier for DP training')
    parser.add_argument('--target_epsilon', type=float, default=32, help='Target epsilon for differential privacy')
    parser.add_argument('--transforms_name', type=str, default=None, help='Augmentations to use')

    # Logging and setup etc
    parser.add_argument('--save_model', type=str, default=None, help='Whether to save the model')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_phys_batchsize', type=int, default=1500, help='how many samples fit on the device per step')
    parser.add_argument('--param_setter', type=str, default="classifier_and_pooler", help='Dataset name to use')

    # Model and Dataset
    parser.add_argument('--experiment_name', type=str, default="untitled", help='Name of the experiment')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased", help='Model name to use')
    parser.add_argument('--dataset_name', type=str, default="sst2", help='Dataset name to use')
    parser.add_argument('--glue', type=bool, default=True, help='Whether the dataset is from HF glue')

    # Parse the arguments
    args = parser.parse_args()

    return args


def commandline():
    args = parse_args()

    # ------------- Logging ----------------------    
    wandb.init(project=args.dataset_name, name=args.experiment_name)

    wandb.config.update(vars(args))   

    trainable_param_setter = get_param_setter_from_str(args.param_setter)
    transform_list = get_transforms_from_str(args.transforms_name)
    K = len(transform_list)


    # Call the function with parsed arguments
    run(
        dataset_name=args.dataset_name,
        transform_list=transform_list,
        K=K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        trainable_param_setter=trainable_param_setter,
        num_labels=args.num_labels,
        noise_multiplier=args.noise_multiplier,
        dataset_size=args.dataset_size,
        save_model=args.save_model,
        target_epsilon=args.target_epsilon,
        early_stop_patience=args.early_stop_patience,
        model_name=args.model_name,
        glue=args.glue,
        max_phys_batchsize=args.max_phys_batchsize
    )

    wandb.finish()


def debug(params):

    wandb.init(project=params['dataset_name'], name="debug")
    wandb.config.update(params)   

    trainable_param_setter = get_param_setter_from_str("classifier_and_pooler")
    transform_list = get_transforms_from_str(params["transform_name"])
    del params["transform_name"]
    K = len(transform_list)

    params.update({"K":K,"transform_list":transform_list,"trainable_param_setter":trainable_param_setter})

    run(**params)


if __name__ == "__main__":
    commandline()


