import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from opacus.utils.batch_memory_manager import BatchMemoryManager
import wandb
from util.logging_util import track_time


def train(
    dp_model,
    dp_train_loader,
    dp_optimizer,
    device,
    K,
    max_phys_batch_size,
    steps,
    task_name,
):
    

    logging_interval = 10
    to_mean_metrics = []
    to_mean_losses = []
    dp_model.train()
    
    # Reshape (N, K, max_len) to (N*K, max_len)
    reshape_flatten = lambda x: x.view(-1, x.size(-1))  

    augmentation_max_physical_batchsize = max_phys_batch_size // K

    # Time Logging
    data_loading_times, forward_times, backward_times, optimizer_times = {"this_batch":[],"log":[]},{"this_batch":[],"log":[]},{"this_batch":[],"log":[]},{"this_batch":[],"log":[]},

    step = steps
    # Using BatchMemoryManager for large batches
    with BatchMemoryManager(data_loader=dp_train_loader, max_physical_batch_size=augmentation_max_physical_batchsize, optimizer=dp_optimizer) as memory_safe_data_loader: 
        start_time = track_time()
        for batch in memory_safe_data_loader:

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
            dp_optimizer.zero_grad(set_to_none=True) 

            outputs = dp_model(**batch)

            loss = outputs.loss 

            metric = compute_task_metric(outputs,batch,task_name)

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
                

# Define task-specific metric functions
def compute_accuracy(preds, labels):
    return accuracy_score(labels, preds)

def compute_mcc(preds, labels):
    return matthews_corrcoef(labels, preds)

def compute_f1_ner(preds, labels):
    return f1_score(labels, preds, average='weighted')


def compute_task_metric(outputs, batch, task):
    # Get predictions and labels
    if task != 'regression':  # For classification tasks
        preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
    else:  # For regression (not in current tasks, but flexibility for future)
        preds = outputs.logits.detach().cpu().numpy()
        
    labels = batch['labels'].detach().cpu().numpy()

    # Map tasks to corresponding metric functions
    task_metrics = {
    'cola': compute_mcc,  # CoLA - Matthews Correlation Coefficient (MCC)
    'qnli': compute_accuracy,  # QNLI - Accuracy
    'sst2': compute_accuracy,  # SST-2 - Accuracy
    'conll2003': compute_f1_ner  # CoNLL 2003 (NER) - F1 Score
    }

    # Select the appropriate metric function based on the task
    metric_fn = task_metrics.get(task.lower(), compute_accuracy)  # Default to accuracy if task not listed
    
    # Compute and return the metric value
    metric_value = metric_fn(preds, labels)
    
    return metric_value
