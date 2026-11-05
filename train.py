import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from opacus.utils.batch_memory_manager import BatchMemoryManager
import wandb
from util.logging_util import track_time


def _new_timers():
    return {"this_batch": [], "log": []}


def train(
    dp_model,
    dp_train_loader,
    dp_optimizer,
    device,
    K,
    max_phys_batch_size,
    steps,
    task_name,
    multi_label=False,
    metric=None,
):


    logging_interval = 10
    to_mean_metrics = []
    to_mean_losses = []
    dp_model.train()

    # Reshape (N, K, max_len) to (N*K, max_len)
    reshape_flatten = lambda x: x.view(-1, x.size(-1))

    augmentation_max_physical_batchsize = max(1, max_phys_batch_size // K)

    # Time Logging
    timers = {name: _new_timers() for name in ("data_loading", "forward", "backward", "optimizer")}

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

            start_time = track_time(timers["data_loading"],start_time)

            # Forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            if multi_label: batch['labels'] = batch['labels'].float()
            dp_optimizer.zero_grad(set_to_none=True)

            outputs = dp_model(**batch)

            loss = outputs.loss

            batch_metric = compute_task_metric(outputs,batch,task_name,multi_label=multi_label,metric=metric)

            start_time = track_time(timers["forward"],start_time)

            # Backward pass
            loss.backward()
            start_time = track_time(timers["backward"],start_time)

            # Optimization
            dp_optimizer.step()
            start_time = track_time(timers["optimizer"],start_time)


            # Logging loss and acc
            is_updated = not (dp_optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch without incrementing the count.
            if is_updated:
                step += 1

                for times in timers.values():
                    times["log"].append(sum(times["this_batch"]))
                    times["this_batch"] = []

                # NOTE: metric/loss of the last physical batch of the logical batch, over all K views
                to_mean_metrics.append(batch_metric)
                to_mean_losses.append(loss.item())
                if step % logging_interval == logging_interval-1:
                    l, m = np.mean(to_mean_losses), np.mean(to_mean_metrics)
                    stats = {
                        "train_loss": l,
                        "train_metric": m,
                        "optimizer_time": np.mean(timers["optimizer"]["log"]),
                        "forward_time": np.mean(timers["forward"]["log"]),
                        "backward_time": np.mean(timers["backward"]["log"]),
                        "data_loading_time": np.mean(timers["data_loading"]["log"]),
                    }
                    wandb.log(stats, step=step)

                    # Reset the timers and accumulators (in place; rebinding a loop variable does nothing)
                    for times in timers.values():
                        times["this_batch"].clear()
                        times["log"].clear()
                    to_mean_metrics, to_mean_losses = [], []

    return step



def eval(model, eval_loader,device,task_name,multi_label=False,metric=None):
    """
    Evaluate on a (non-augmented) split. The metric is computed over the whole split
    (not averaged over batches); the loss is the example-weighted mean.
    """
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in eval_loader:

            batch = {k: v.to(device) for k, v in batch.items()}
            if multi_label: batch['labels'] = batch['labels'].float()
            outputs = model(**batch)

            bs = batch['labels'].shape[0]
            total_loss += outputs.loss.item() * bs
            n += bs
            all_logits.append(outputs.logits.detach().cpu())
            all_labels.append(batch['labels'].detach().cpu())

    model.train()

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metric_value = compute_metric(logits, labels, task_name, multi_label=multi_label, metric=metric)
    return (metric_value, total_loss / max(n, 1))


# Define task-specific metric functions
def compute_accuracy(preds, labels):
    return accuracy_score(labels, preds)

def compute_mcc(preds, labels):
    return matthews_corrcoef(labels, preds)

def compute_f1_weighted(preds, labels):
    return f1_score(labels, preds, average='weighted')

def compute_micro_f1(preds, labels):
    return f1_score(labels, preds, average='micro', zero_division=0)

def compute_macro_f1(preds, labels):
    return f1_score(labels, preds, average='macro', zero_division=0)


METRICS = {
    'accuracy': compute_accuracy,
    'mcc': compute_mcc,
    'f1_weighted': compute_f1_weighted,
    'micro_f1': compute_micro_f1,
    'macro_f1': compute_macro_f1,
}

# Map tasks to their default metric
TASK_METRICS = {
    'cola': 'mcc',  # CoLA - Matthews Correlation Coefficient (MCC)
    'qnli': 'accuracy',  # QNLI - Accuracy
    'sst2': 'accuracy',  # SST-2 - Accuracy
    'conll2003': 'f1_weighted',  # CoNLL 2003 (NER) - F1 Score
}


def default_metric_name(task, multi_label=False):
    if multi_label:
        return 'micro_f1'
    return TASK_METRICS.get(task.lower(), 'accuracy')  # Default to accuracy if task not listed


def compute_metric(logits, labels, task, multi_label=False, metric=None):
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().numpy()
    if multi_label:
        preds = (torch.sigmoid(logits) > 0.5).int().numpy()
        labels = labels.astype(int)
    elif task == 'regression':  # For regression (not in current tasks, but flexibility for future)
        preds = logits.numpy()
    else:  # For classification tasks
        preds = np.argmax(logits.numpy(), axis=1)
    metric_fn = METRICS[metric or default_metric_name(task, multi_label)]
    return metric_fn(preds, labels)


def compute_task_metric(outputs, batch, task, multi_label=False, metric=None):
    return compute_metric(outputs.logits, batch['labels'], task, multi_label=multi_label, metric=metric)
