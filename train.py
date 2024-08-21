import torch.nn as nn
import torch
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager

MAX_PHYSICAL_BATCH_SIZE = 32
LOGGING_INTERVAL = 1

def train(
    dp_model,
    dp_train_loader,
    dp_optimizer,
    device,
    K,
):
    dp_model.train()
    
    # Reshape (N, K, max_len) to (N*K, max_len)
    reshape_flatten = lambda x: x.view(-1, x.size(-1))  

    # Using BatchMemoryManager for large batches
    with BatchMemoryManager(data_loader=dp_train_loader, max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, optimizer=dp_optimizer) as memory_safe_data_loader: 
        for step, batch in enumerate(memory_safe_data_loader):
            print(f"\n NEW BATCH" + "=" * 100 + "\n")
            print(f"MultiView Sentence (tokens) shape: {batch['input_ids'].shape}\nshould be: [B, K, max_len=128]")
            
            dp_optimizer.zero_grad(set_to_none=True) 

            if K > 1:
                for column in batch:
                    print(f"batchpart: {column}")

                    if column == 'labels':
                        # Duplicate labels from (N) to (N*K) relating to their samples and augmented versions
                        batch['labels'] = torch.repeat_interleave(batch['labels'], repeats=K, dim=0)
                        print(f"labels: {batch['labels']}")
                    
                    else: 
                        batch[column] = reshape_flatten(batch[column])

                print(f"MultiView Sentence (tokens) after reshape: {batch['input_ids'].shape}\nshould be: [B*K, max_len=128]")

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = dp_model(**batch)

            #logits = outputs.logits
            loss = outputs[0]

            # Backward pass and optimization
            loss.backward()
            dp_optimizer.step()

            # Logging or monitoring loss
            if step % LOGGING_INTERVAL == 0:
                print(f"Step {step}, Loss: {loss.item()}")
                
