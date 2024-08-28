import json
import time
import torch.nn as nn
import torch
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager


def train(
    dp_model,
    dp_train_loader,
    dp_optimizer,
    device,
    K,
    logger,
    max_phys_batch_size = 200,
    logging_interval = 100,
):
    
    to_mean_accuracies = []
    to_mean_losses = []
    dp_model.train()
    
    # Reshape (N, K, max_len) to (N*K, max_len)
    reshape_flatten = lambda x: x.view(-1, x.size(-1))  

    augmentation_max_physical_batchsize = max_phys_batch_size // K

    # Time Logging
    data_loading_times, forward_times, backward_times, optimizer_times = [], [], [], []
    optimizer_time = time.time()
    
    # Using BatchMemoryManager for large batches
    with BatchMemoryManager(data_loader=dp_train_loader, max_physical_batch_size=augmentation_max_physical_batchsize, optimizer=dp_optimizer) as memory_safe_data_loader: 
        for step, batch in enumerate(memory_safe_data_loader):

            dp_optimizer.zero_grad(set_to_none=True) 

            # reshape batch flatten
            for column in batch:
                if column == 'labels':
                    # Duplicate labels from (N) to (N*K) relating to their samples and augmented versions
                    batch['labels'] = torch.repeat_interleave(batch['labels'], repeats=K, dim=0)                    
                else: 
                    batch[column] = reshape_flatten(batch[column])
        
            start = time.time()
            data_loading_times.append(start-optimizer_time)


            # Forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = dp_model(**batch)

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
            dp_optimizer.step()
            optimizer_time = time.time()
            optimizer_times.append(optimizer_time-backward_time)

            # Logging loss and acc
            to_mean_accuracies.append(acc)
            to_mean_losses.append(loss.item())
            if step % logging_interval == 0:
                l,a = np.mean(to_mean_losses),np.mean(to_mean_accuracies)
                print(f"Step {step}, Loss: {l}, Accuracy: {a}")
                logger.info(f"Step {step}, Loss: {l}, Accuracy: {a}" + json.dumps({"optimizer_time":np.mean(optimizer_times),
                                    "forward_time":np.mean(forward_times),
                                    "backward_time":np.mean(backward_times),
                                    "data_loading_time":np.mean(data_loading_times),}))
                optimizer_times,forward_times, backward_times, data_loading_times = [],[],[],[]
                to_mean_accuracies,to_mean_losses = [],[]
                
