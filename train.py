import torch
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from util.logging_util import track_time, log_metrics


def train(
    dp_model,
    dp_train_loader,
    dp_optimizer,
    device,
    K,
    logger,
    max_phys_batch_size,
    logs_per_epoch = 10,
):
    
    logging_interval = len(dp_train_loader) // logs_per_epoch
    if logging_interval==0: logging_interval=1

    
    to_mean_accuracies = []
    to_mean_losses = []
    dp_model.train()
    
    # Reshape (N, K, max_len) to (N*K, max_len)
    reshape_flatten = lambda x: x.view(-1, x.size(-1))  

    augmentation_max_physical_batchsize = max_phys_batch_size // K

    # Time Logging
    data_loading_times, forward_times, backward_times, optimizer_times = [], [], [], []
    start_time = track_time()
    step = 0
    # Using BatchMemoryManager for large batches
    with BatchMemoryManager(data_loader=dp_train_loader, max_physical_batch_size=augmentation_max_physical_batchsize, optimizer=dp_optimizer) as memory_safe_data_loader: 
        for batch in memory_safe_data_loader:

            dp_optimizer.zero_grad(set_to_none=True) 

            # reshape batch flatten
            for column in batch:
                if column == 'labels':
                    # Duplicate labels from (N) to (N*K) relating to their samples and augmented versions
                    batch['labels'] = torch.repeat_interleave(batch['labels'], repeats=K, dim=0)                    
                else: 
                    batch[column] = reshape_flatten(batch[column])
        
            start_time = track_time(data_loading_times,start_time)

            # Forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = dp_model(**batch)

            loss = outputs.loss 
            preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
            labels = batch['labels'].detach().cpu().numpy()
            acc = (preds==labels).mean()

            start_time = track_time(forward_times,start_time)
            
            # Backward pass
            loss.backward()
            start_time = track_time(backward_times,start_time)

            # Optimization
            dp_optimizer.step()
            start_time = track_time(optimizer_times,start_time)


            # Logging loss and acc
            is_updated = not (dp_optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch without incrementing the count.
            if is_updated: 
                step += 1  

                to_mean_accuracies.append(acc)
                to_mean_losses.append(loss.item())
                if step % logging_interval == 0:
                    l, a = np.mean(to_mean_losses), np.mean(to_mean_accuracies)
                    metrics = {
                        "train_loss": l,
                        "train_accuracy": a,
                        "optimizer_time": np.mean(optimizer_times),
                        "forward_time": np.mean(forward_times),
                        "backward_time": np.mean(backward_times),
                        "data_loading_time": np.mean(data_loading_times),
                    }
                    log_metrics(step, metrics, logger)
                    
                    # Reset the timers and accumulators
                    optimizer_times, forward_times, backward_times, data_loading_times = [], [], [], []
                    to_mean_accuracies, to_mean_losses = [], []



def eval(model, eval_loader,device):
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

    model.train()

    return (test_top1_avg,losses_avg)
                
