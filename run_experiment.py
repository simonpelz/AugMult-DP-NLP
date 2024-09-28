import argparse
import types
import os
from uu import Error

import torch
import torch.optim as optim

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from augmult.augmentations import get_transforms_from_str
from train import train, eval
from augmult.data import dp_dataloader, init_mv_collate, non_dp_tokenize_dataloader, get_dataset
from util.early_stopper import EarlyStopping
from util.privacy_engine_util import _prepare_model_modified, empty_batch_handling, prepare_gradsamplers
from util.logging_util import aug_name
from util.different_finetune_modes import get_param_setter_from_str, model_and_tokenizer

import wandb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')

# Disable parallelism for tokenizers necessary for backtranslation
#os.environ["TOKENIZERS_PARALLELISM"] = "false" # TODO move to augmentation class in trslt=True


def run(dataset_name, transform_list, K, epochs, batch_size, lr, max_grad_norm, trainable_param_setter, num_labels, noise_multiplier=None, dataset_size=None, save_model=None, target_epsilon=None, early_stop_patience=10, model_name = "bert-base-uncased", glue=True, logs_per_epoch=10,max_phys_batchsize=1500,total_steps=None):    

    # ---------------- Initialisation ------------------
    
    # Model, Tokenizer
    model, tokenizer = model_and_tokenizer(model_name, num_labels)
    total_p, trainable_p = trainable_param_setter(model)
    finetune_percent = trainable_p / total_p * 100

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Dataloaders
    dataset = get_dataset(dataset_name, glue = glue)
    mv_train_loader = dp_dataloader(dataset["train"],dataset_size,tokenizer,transform_list,batch_size)
    valid_loader = non_dp_tokenize_dataloader(dataset['validation'], tokenizer, max_phys_batchsize)

    steps_per_epoch = len(mv_train_loader)
    if epochs is None:
        if total_steps is None: raise ValueError
        epochs = total_steps//steps_per_epoch
        early_stop_patience = max(4,(epochs//3))

    # Logging
    wandb.config.update({"early_stop_patience":early_stop_patience,"epochs":epochs,'K': K,'batches per epoch': steps_per_epoch,
        "transform_list": aug_name(transform_list),"finetune_percent":finetune_percent,})
    
    # ------------- Make DP with AugMult -----------------

    delta = 1 / len(mv_train_loader.dataset)
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
    
    # Override the empty batch shapes provided by privacy engine
    mv_collate = init_mv_collate(tokenizer,transform_list,max_length=128)
    empty_batch_handling(mv_train_loader=mv_train_loader,dp_train_loader=dp_train_loader,mv_collate=mv_collate)

    # Custom Grad Samplers
    dp_model = prepare_gradsamplers(K,dp_model)

    # -------------- Training ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_model.to(device=device)

    train_inputs = {
        'dp_model': dp_model,
        'task_name':dataset_name,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': K,
        'logs_per_epoch': logs_per_epoch,
        'max_phys_batch_size': max_phys_batchsize,
        'steps': 0,
    }

    valid_metric = 0
    for i in range(epochs):
        # Train for 1 epoch
        train_inputs['steps'] = train(**train_inputs)
        # Evaluate
        valid_metric, valid_loss = eval(dp_model, valid_loader,task_name=dataset_name, device=device)
        wandb.log({"epoch": i+1,"valid_metric": valid_metric,"valid_loss": valid_loss})
        # Stop early if stagnant
        if early_stop.step(valid_loss): 
            wandb.log({"Stopped early": True})
            break


    # ------------ Evaluation ---------------------

    # Privacy accounting
    try:
        real_eps = privacy_engine.accountant.get_epsilon(delta)
    except Error as error: # TODO what was the exact error? something infinity
        print(error)
        real_eps = float('inf')
    wandb.log({"epsilon": real_eps, "delta":delta})

    # Checkpoint saving
    if save_model is not None:
        torch.save({
            'metric': valid_metric,
            'model_state_dict': dp_model.state_dict(),
            'epoch': epochs # TODO stop early handling
        }, f"./logs_and_ckpts/ckpts/{save_model}")
        print(f"saved to:./logs_and_ckpts/ckpts/{save_model}")

    # TODO is this smart? edgecase epoch=0 etc. returning for sweep setup
    return valid_metric


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
    parser.add_argument('--logs_per_epoch', type=int, default=10, help='how often to log per epoch')
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
    wandb.init(project=args.dataset_name, dir="./logs_and_ckpts/wandb", name=args.experiment_name)

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
        logs_per_epoch=args.logs_per_epoch,
        max_phys_batchsize=args.max_phys_batchsize
    )

    wandb.finish()


def debug(params):

    wandb.init(project=params['dataset_name'], dir="./logs_and_ckpts/wandb", name="debug")
    wandb.config.update(params)   

    trainable_param_setter = get_param_setter_from_str("classifier_and_pooler")
    transform_list = get_transforms_from_str(params["transform_name"])
    del params["transform_name"]
    K = len(transform_list)

    params.update({"K":K,"transform_list":transform_list,"trainable_param_setter":trainable_param_setter})

    run(**params)


if __name__ == "__main__":
    commandline()


