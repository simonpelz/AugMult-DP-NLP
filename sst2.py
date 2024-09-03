import logging
import types
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from datasets import load_dataset

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.data_loader import shape_safe, dtype_safe

from train import train, eval
from augmult.data import MultiViewTextDataset, non_dp_tokenize_Dataloader
from util.privacy_engine_util import _prepare_model_modified, dict_wrap_collate_with_empty
from augmult.augmented_grad_samplers import AugmentationMultiplicity
from util.logging_util import log_from_dict
from util.different_finetune_modes import model_and_tokenizer

import wandb

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model Information
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2

# Differential privacy parameters
EPSILON = 16.0

# Environment
LOGS_PER_EPOCH = 10
MAX_PHYSICAL_BATCH_SIZE = 512


def sst2(transform_list, epochs, batch_size, lr, max_grad_norm, noise_multiplier, logger, trainable_param_setter, dataset_size=None, save_model=None, experiment_name = "untitled"):    

    # ----------- Initialisation -------------

    # Model, Optimizer, Tokenizer
    model, tokenizer = model_and_tokenizer(MODEL_NAME, NUM_LABELS)
    total_p, trainable_p = trainable_param_setter(model)
    logger.info(f"total params: {total_p}, trainable:{trainable_p}")

    optimizer = optim.SGD(model.parameters(), lr=lr)

    if not os.environ["TOKENIZERS_PARALLELISM"]:
        logger.info(f"Tokenizer parallelism turned off")

    # Augmentations K
    K = len(transform_list)

    # Dataloaders
    dataset = load_dataset("glue", "sst2")

    if dataset_size is not None:
        modified_trainset = dataset['train'].select(range(dataset_size))
    else:
        modified_trainset = dataset['train']
    mv_train_set = MultiViewTextDataset(modified_trainset, tokenizer, transform_list=transform_list)
    mv_train_loader = mv_train_set.__dataloader__(batch_size)

    valid_loader = non_dp_tokenize_Dataloader(dataset['validation'], tokenizer, batch_size * K)

    # GPU handling
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")

    # ------------- Logging ----------------------
    
    wandb.init(project=experiment_name)

    # Log hyperparameters
    hyperparams = {
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "epochs": epochs,
        "batch_size": batch_size,
        'K': K,
        'batches per epoch': len(mv_train_loader),
        "learning_rate": lr,
        "max_grad_norm": max_grad_norm,
        "noise_multiplier": noise_multiplier,
        "transform_list": transform_list,
        "dataset_size": dataset_size
    }
    log_from_dict(logger, hyperparams)
    wandb.config.update(hyperparams)   

    # ------------- Make DP with AugMult -----------------

    # Ensure DP compatibility
    model.train()
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()

    # Hack to load the custom AugmultGradSamplerModule
    privacy_engine._prepare_model = types.MethodType(_prepare_model_modified, privacy_engine)

    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=mv_train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="bias_only",
    )

    # Override the empty batch shapes provided by privacy engine
    sample_empty_shapes = {k: (0, *shape_safe(v)) for k, v in mv_train_loader.dataset[0].items()}
    dtypes = {k: dtype_safe(v) for k, v in mv_train_loader.dataset[0].items()}
    dp_train_loader.collate_fn = dict_wrap_collate_with_empty(collate_fn=default_collate, sample_empty_shapes=sample_empty_shapes, dtypes=dtypes)

    # Custom Grad Samplers
    dp_model.K = K
    augmented = AugmentationMultiplicity(K)
    dp_model.GRAD_SAMPLERS[nn.Linear] = augmented.compute_linear_grad_sample
    dp_model.GRAD_SAMPLERS[nn.LayerNorm] = augmented.compute_layer_norm_grad_sample

    dp_model.to(device=device)

    # -------------- Training ----------------------
    train_inputs = {
        'dp_model': dp_model,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': K,
        'logger': logger,
        'logs_per_epoch': LOGS_PER_EPOCH,
        'max_phys_batch_size': MAX_PHYSICAL_BATCH_SIZE,
    }

    for i in range(epochs):
        logger.info(f"Epoch {i+1} starting.")
        train(**train_inputs)

        valid_acc, valid_loss = eval(dp_model, valid_loader, device=device)
        logger.info(f"Validation: acc:{valid_acc}, loss:{valid_loss}")

        wandb.log({
            "epoch": i+1,
            "valid_accuracy": valid_acc,
            "valid_loss": valid_loss
        })


    # ------------ Evaluation ---------------------
    delta = 1 / (dataset_size if dataset_size is not None else len(dataset['train']))
    real_eps = privacy_engine.accountant.get_epsilon(delta)
    logger.info(f"accountant epsilon: {real_eps}")
    wandb.log({"epsilon": real_eps,
               "delta":delta})

    if save_model is not None:
        torch.save({
            'acc': valid_acc,
            'model_state_dict': dp_model.state_dict(),
        }, f"./ckpts/{save_model}")
        logger.info(f"saved to:./logs_and_ckpts/ckpts/{save_model}")

    logger.info("=" * 50)
    wandb.finish()

    # TODO Multi-GPU
    # TODO slurm?
