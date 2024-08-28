from multiprocessing import get_logger
import types

import torch
import torch.optim as optim
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from train import train
from data import MultiViewTextDataset, prepare_eval_dataloaders
from privacy_engine_util import _prepare_model_modified
from augmentations import Augmentations
from augmented_grad_samplers import AugmentationMultiplicity
from logging_util import log_from_dict

# TODO is this the only way?
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Model Information
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2

# Differential privacy parameters
EPSILON = 16.0


#Environment
MAX_PHYSICAL_BATCH_SIZE = 150

def model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model, tokenizer


def sst2(transform_list,epochs,batch_size,lr,dataset_size,max_grad_norm,noise_multiplier,logger):
    
    # ----------- Initialisation -------------

    # Logger
    #logger = get_file_logger("first test", "training.log")

    # Model
    model, tokenizer = model_and_tokenizer(MODEL_NAME, NUM_LABELS)
    # TODO Fix embedding grad sampler and unfreeze?
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    logger.info("Embedding layers are frozen")
    if not os.environ["TOKENIZERS_PARALLELISM"]: logger.info(f"Tokenizer paralellism turned off")

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Augmentation types
    K = len(transform_list)
    # Dataloaders
    pre_dataset = load_dataset("glue", "sst2")
    #TODO remove: Use only a subset of the training dataset for testing
    subset_train_dataset = pre_dataset["train"].select(range(dataset_size))
    #subset_train_dataset = pre_dataset["train"]
    mv_train_loader = MultiViewTextDataset(subset_train_dataset, tokenizer, transform_list=transform_list).__dataloader__(batch_size)
    #valid_loader, test_loader = prepare_eval_dataloaders(pre_dataset, tokenizer, EVAL_BATCH_SIZE=64)

    # GPU handling
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)
    print(device_name)

    # ------------- Make DP with AugMult-----------------

    # Ensure DP compatibility
    model.train()
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()
    # A Hack to load the custom AugmultGradSamplerModule. Methods almost identical, 
    privacy_engine._prepare_model = types.MethodType(_prepare_model_modified, privacy_engine)

    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=mv_train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="augmult",
        #target_epsilon=EPSILON,
        #target_delta=DELTA,
        #epochs=EPOCHS,
    )

    # Custom Grad Samplers
    dp_model.K = K
    augmented = AugmentationMultiplicity(K)
    dp_model.GRAD_SAMPLERS[nn.Linear] = augmented.compute_linear_grad_sample
    dp_model.GRAD_SAMPLERS[nn.LayerNorm] = augmented.compute_layer_norm_grad_sample
    # dp_model.GRAD_SAMPLERS[nn.Embedding] = augmented.compute_embedding_grad_sample # TODO fix embedding grad smapler?

    # ------------- Logging ----------------------
    d = {'K':K,'Batchsize':batch_size,'transform_list':transform_list,'epochs':epochs, 'max_grad_norm':max_grad_norm, 'noise_mult': noise_multiplier,'batches per epoch':len(dp_train_loader)}
    log_from_dict(logger,d)

    # -------------- Training ----------------------
    train_inputs = {
        'dp_model': dp_model,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': K,
        'logger': logger,
        'logging_interval': 25,
        'max_phys_batch_size': MAX_PHYSICAL_BATCH_SIZE,
        }
    
    for i in range(epochs):
        logger.info(f"Epoch {i+1} starting.")
        train(**train_inputs)

    real_eps = privacy_engine.accountant.get_epsilon(1/dataset_size)    
    logger.info(f"accountant epsilon: {real_eps}")
    logger.info("="*50)
    # TODO add validation
    # TODO add logging
    # TODO slurm?
    # TODO add save ckpt
