from gettext import dpgettext
import json
import torch
import wandb
from augmult.augmentations import Augmentations
from util.logging_util import get_file_logger, aug_name
from sst2 import sst2
from util.different_finetune_modes import *


def safe_sst2(**kwargs):
    try:
        sst2(**kwargs)
    except torch.OutOfMemoryError as e:
        wandb.log({"Cuda OOM": True})

FAST_K1 = [lambda x: x]

def get_defaults():

    dp_params={ 'max_grad_norm': 1.0,
                'target_epsilon': 64,
                'noise_multiplier': None,
                'transform_list': FAST_K1,
                }

    log_params={'save_model': None,
                'experiment_name': None,
                }

    training_params={
                    'num_labels': 2,
                    'epochs': 6,
                    "early_stop_patience": 4,
                    'batch_size': 500,
                    'lr': 0.01,
                    'dataset_size': 10000,
                    'trainable_param_setter': classifier_and_pooler,
                }
    return  {**dp_params,**log_params, **training_params}

def sweep_lr():
    params = get_defaults()
    s = f"lr"
    logger = get_file_logger(s, "lr.log")
    params.update({'experiment_name': s,
                'logger': logger,})

    for lr in [0.0001,0.001,0.01,0.05,0.1,0.5]:
        params['lr'] = lr
        safe_sst2(**params)



def augmentation_times():
    s = f"Augmentations time and performance"
    logger = get_file_logger(s, "Augmentation_comparison.log")

    params = {
                'epochs': 1,
                'batch_size': 500,
                'lr': 0.05,
                'dataset_size': 1500,
                'max_grad_norm': 1.0,
                'target_epsilon': 32,
                #'noise_multiplier': 0.1,
                #'save_model': None,
                'experiment_name': s,
                'logger': logger,
                'trainable_param_setter': classifier_and_pooler,
                'grad_sample_mode': "augmult"
            }
    
    a = Augmentations()
    s = a.single_aug
    all_augs_separately = [s(a.unaugmented),s(a.context_replacement),s(a.context_insert),s(a.typo),
                           s(a.del_word),s(a.swap_char),s(a.swap_word)] #,s(a.back_translate)

    for t_list in all_augs_separately:
        logger.info(f"{aug_name(t_list[1])}")
        safe_sst2(**params ,transform_list=t_list)
        logger.info(f"\n\n")



def test_working():
    s = f"testing"
    logger = get_file_logger(s, "scrap.log")

    params = {
                'epochs': 2,
                'batch_size': 500,
                'lr': 0.05,
                'dataset_size': 1000,
                'max_grad_norm': 1.0,
                'target_epsilon': 8,
                #'noise_multiplier': 0.1,
                #'save_model': None,
                'experiment_name': s,
                'logger': logger,
                'trainable_param_setter': classifier_and_pooler,
                'grad_sample_mode': "augmult",
                'transform_list': Augmentations().special_blend_no_reasoning(),

            }
    safe_sst2(**params)


def dp_K1():
    hypparams=get_defaults()
    params = {
            'max_grad_norm': 1.0,
            'target_epsilon': 32,
            'dataset_size': None,
            'epochs':6,
            'lr':1,
        }
    hypparams.update(params)

    for tl in [Augmentations(trnslt=False).eda_changed()]:
        name = f"Full | K:{len(tl)}"
        hypparams.update({"experiment_name": name,'transform_list':tl,})
        safe_sst2(**hypparams)
    

def non_dp_K1():
    hypparams=get_defaults()
    params = {
            'max_grad_norm': 1.0,
            'target_epsilon': 32,
            'dataset_size': None,
        }
    hypparams.update(params)

    for lr in [0.1,1]:
        name = f"Full | lr:{lr}, norm:{params['max_grad_norm']},eps:{params['target_epsilon']}"
        hypparams.update({"lr":lr,"experiment_name": name})
        safe_sst2(**hypparams)

    

def realistic_eps():
    hypparams=get_defaults()
    params = {
            'max_grad_norm': 1.0,
            'target_epsilon': 32,
            'dataset_size': None,
            'epochs':15,
            'lr':1,
            'batchsize':2500
        }
    hypparams.update(params)

    name = f"Full larger batchsize | K:1"
    hypparams.update({"experiment_name": name,'transform_list':FAST_K1,})
    safe_sst2(**hypparams)
    


def main():
    realistic_eps()

if __name__ == "__main__":
    main()
