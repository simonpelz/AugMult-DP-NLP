import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import torch
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset

from train import train, eval
from data import non_dp_tokenize_Dataloader
from logging_util import log_from_dict
from logging_util import get_file_logger

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Model Information
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2


#Environment
LOGS_PER_EPOCH = 10


def non_dp_sst2(epochs,batch_size,lr,logger,dataset_size=None,only_classifier=True, save_model=None):
    
    # ----------- Initialisation -------------

    # Model, Optimizer, Tokenizer
    model, tokenizer = model_and_tokenizer(MODEL_NAME, NUM_LABELS,only_classifier=only_classifier)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if not os.environ["TOKENIZERS_PARALLELISM"]: logger.info(f"Tokenizer paralellism turned off")

    # Dataloaders
    dataset = load_dataset("glue", "sst2")
    # Tokenization of the datasets
    if dataset_size is not None: modified_trainset = dataset['train'].select(range(dataset_size))
    else: modified_trainset = dataset['train']
    tokens_train = modified_trainset.map(lambda x: tokenizer(x['sentence'], max_length=128, padding='max_length', truncation=True), batched=True)
    tokens_valid = dataset['validation'].map(lambda x: tokenizer(x['sentence'], max_length=128, padding='max_length', truncation=True), batched=True)
    tokens_train = tokens_train.remove_columns(['idx','sentence']).rename_column("label", "labels") 
    tokens_valid = tokens_valid.remove_columns(['idx','sentence']).rename_column("label", "labels") 
    # Set the format to PyTorch tensors
    tokens_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokens_valid.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    valid_loader = DataLoader(tokens_valid, shuffle=False, batch_size=batch_size)
    train_loader = DataLoader(tokens_train, shuffle=False, batch_size=batch_size)

    # GPU handling
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)
    logging.info(f"Using device: {device}")

    model.train()
   
    # ------------- Logging ----------------------
    d = {'Batchsize':batch_size,'epochs':epochs,'batches per epoch':len(train_loader)}
    log_from_dict(logger,d)

    # -------------- Training ----------------------
    train_inputs = {
        'model': model,
        'train_loader': train_loader,
        'optimizer': optimizer,
        'device': device,
        'logger': logger,
        'logs_per_epoch': LOGS_PER_EPOCH,
        }
    
    for i in range(epochs):
        logger.info(f"Epoch {i+1} starting.")
        train(**train_inputs)

    valid_acc, valid_loss = eval(model, valid_loader, logger = logger, device=device)

    if save_model is not None: 
        torch.save({
             'acc':valid_acc,
             'model_state_dict': model.state_dict(),
        },f"./ckpts/{save_model}")
        logger.info(f"saved to:./ckpts/{save_model}")

    logger.info("="*50)


def train(
    model,
    train_loader,
    optimizer,
    device,
    logger,
    logs_per_epoch = 10,
):
    
    logging_interval = len(train_loader) // logs_per_epoch
    if logging_interval==0:logging_interval=1
    
    to_mean_accuracies = []
    to_mean_losses = []
    model.train()

    # Time Logging
    data_loading_times, forward_times, backward_times, optimizer_times = [], [], [], []
    optimizer_time = time.time()
    step = 0

    for batch in train_loader:

        optimizer.zero_grad() 
    
        start = time.time()
        data_loading_times.append(start-optimizer_time)

        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss 
        preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
        labels = batch['labels'].detach().cpu().numpy()
        acc = (preds==labels).mean()

        forward_time = time.time()
        forward_times.append(forward_time -start)
        
        # Backward pass
        loss.backward()
        backward_time = time.time()
        backward_times.append(backward_time -forward_time)

        # Optimization
        optimizer.step()
        optimizer_time = time.time()
        optimizer_times.append(optimizer_time-backward_time)

        # Logging loss and acc
        step += 1  

        to_mean_accuracies.append(acc)
        to_mean_losses.append(loss.item())
        if step % logging_interval == 0:
            l,a = np.mean(to_mean_losses),np.mean(to_mean_accuracies)
            print(f"Step {step}, Loss: {l:.3f}, Accuracy: {a:.3f}")
            logger.info(f"Step {step}, Loss: {l:.3f}, Accuracy: {a:.3f}  |   " + json.dumps({"optimizer_time":np.mean(optimizer_times),
                                "forward_time":np.mean(forward_times),
                                "backward_time":np.mean(backward_times),
                                "data_loading_time":np.mean(data_loading_times),}))
            optimizer_times,forward_times, backward_times, data_loading_times = [],[],[],[]
            to_mean_accuracies,to_mean_losses = [],[]


def eval(model, eval_loader,device,logger=None):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    losses = []
    test_top1_acc = []

    with torch.no_grad():
        for batch in eval_loader:

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss 
            preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
            labels = batch['labels'].detach().cpu().numpy()
            acc = (preds==labels).mean()

            losses.append(loss.item())
            test_top1_acc.append(acc)

    test_top1_avg = np.mean(test_top1_acc)
    losses_avg  = np.mean(losses)
    s = (f"\Eval set:"f"Loss: {np.mean(losses_avg):.6f} "f"Acc: {test_top1_avg * 100:.6f} ")
    if logger: logger.info(s)
    else: print(s)
    return (test_top1_avg,losses_avg)
                

def model_and_tokenizer(model_name, num_labels, only_classifier=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    
    trainable_layers = [model.classifier] if only_classifier else [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()
    
    print(f"total params: {total_params}, trainable:{trainable_params}")
    return model, tokenizer


def main():
    logger = get_file_logger("Non DP only classifier head", "non_dp.log")
    params = {
    'epochs': 10,
    'lr': 0.01,
    'batch_size': 256,
    'logger': logger,
    'only_classifier': False,
    'save_model': "full_3layers.ckpt"
    }

    s = f"Non DP - 10 Epochs - datasetsize = full - {params['batch_size']} Batchsize - LR = {params['lr']}, only classifier = {params['only_classifier']}"
    logger.info(s)
    non_dp_sst2(**params)

    return
    # find LR (not conclusive)
    for small_size in [True,False]:
        for lr_loop in [0.001,0.005,0.01,0.05,0.1,0.5]:
            s = f"Non DP - 5 Epochs - datasetsize = {params['dataset_size']} - {params['batch_size']} Batchsize - LR = {lr_loop}, only classifier = {small_size}"
            logger.info(s)
            non_dp_sst2(**params)


if __name__ == "__main__":
    main()