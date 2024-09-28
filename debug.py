import run_experiment

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

params = {
    'dataset_name': "sst2",
    'epochs': 33,  
    'batch_size': 543,  
    'lr': 1.1,  
    'max_grad_norm': 2.4,  
    'num_labels': 2,  
    'noise_multiplier': None,  
    'dataset_size': None,  
    'save_model': None,  
    'target_epsilon': 8,  
    'early_stop_patience': 3,  
    'model_name': "bert-base-uncased",  
    'glue': True,  
    'logs_per_epoch': 10,  
    'max_phys_batchsize': 4096, # 5500 still not out of memory!
    'transform_name': "eda5",  
    }

if __name__ == "__main__":
    run_experiment.debug(params)