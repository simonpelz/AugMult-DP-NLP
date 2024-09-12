import types
import os
from uu import Error

import torch
import torch.optim as optim

from datasets import load_dataset

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from train import train, eval
from augmult.data import dp_dataloader, non_dp_tokenize_dataloader
from util.early_stopper import EarlyStopping
from util.privacy_engine_util import _prepare_model_modified, empty_batch_handling, prepare_gradsamplers
from util.logging_util import aug_name
from util.different_finetune_modes import model_and_tokenizer

import wandb

# Disable parallelism for tokenizers necessary for backtranslation
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Environment
LOGS_PER_EPOCH = 10
MAX_PHYSICAL_BATCH_SIZE = 1500


def sst2(transform_list, epochs, batch_size, lr, max_grad_norm, trainable_param_setter,num_labels, noise_multiplier=None,dataset_size=None, save_model=None, experiment_name = "untitled", target_epsilon=None,early_stop_patience=10,model_name = "bert-base-uncased"):    
    
    # ----------- Initialisation -------------
    hyperparams = locals().copy()

    # Model, Optimizer, Tokenizer
    model, tokenizer = model_and_tokenizer(model_name, num_labels)
    total_p, trainable_p = trainable_param_setter(model)
    finetune_percent = trainable_p / total_p * 100

    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Dataloaders
    dataset = load_dataset("glue", "sst2")

    mv_train_loader = dp_dataloader(dataset["train"],dataset_size,tokenizer,transform_list,batch_size)
    valid_loader = non_dp_tokenize_dataloader(dataset['validation'], tokenizer, batch_size * len(transform_list))

    # ------------- Logging ----------------------

    # Augmentations K
    K = len(transform_list)
    delta = 1 / len(mv_train_loader.dataset)
    early_stop = EarlyStopping(patience=early_stop_patience)

    #TODO remove if true
    assert (dataset_size is None) or (len(mv_train_loader.dataset)==dataset_size)
    
    wandb.init(project="sst2", dir="./logs_and_ckpts/wandb", name=experiment_name)

    # Log hyperparameters
    hyperparams.update({
        'K': K,
        'batches per epoch': len(mv_train_loader),
        "transform_list": aug_name(transform_list),
        "finetune_percent":finetune_percent,
    })

    wandb.config.update(hyperparams)   

    # ------------- Make DP with AugMult -----------------

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
    empty_batch_handling(mv_train_loader=mv_train_loader,dp_train_loader=dp_train_loader)

    # Custom Grad Samplers
    dp_model = prepare_gradsamplers(K,dp_model)

    # -------------- Training ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_model.to(device=device)

    train_inputs = {
        'dp_model': dp_model,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': K,
        'logs_per_epoch': LOGS_PER_EPOCH,
        'max_phys_batch_size': MAX_PHYSICAL_BATCH_SIZE,
        'steps': 0,
    }

    for i in range(epochs):
        steps = train(**train_inputs)
        train_inputs['steps']=steps

        valid_acc, valid_loss = eval(dp_model, valid_loader, device=device)
        wandb.log({"epoch": i+1,"valid_accuracy": valid_acc,"valid_loss": valid_loss})

        if early_stop.step(valid_loss): 
            wandb.log({"Stopped early": True})
            break


    # ------------ Evaluation ---------------------
    try:
        real_eps = privacy_engine.accountant.get_epsilon(delta)
    except Error as error: # TODO what was the exact error? something infinity
        print(error)
        real_eps = float('inf')
    wandb.log({"epsilon": real_eps, "delta":delta})

    if save_model is not None:
        torch.save({
            'acc': valid_acc,
            'model_state_dict': dp_model.state_dict(),
        }, f"./logs_and_ckpts/ckpts/{save_model}")
        print(f"saved to:./logs_and_ckpts/ckpts/{save_model}")

    wandb.finish()