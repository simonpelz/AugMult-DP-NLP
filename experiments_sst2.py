from gettext import dpgettext
import json
import torch
from augmult.augmentations import Augmentations
from util.logging_util import get_file_logger, aug_name
from sst2 import sst2
from util.different_finetune_modes import *


def safe_sst2(logger,**kwargs):
    try:
        sst2(**kwargs,logger=logger)
    except torch.OutOfMemoryError as e:
        logger.error(e)

FAST_K1 = [lambda x: x]

def get_defaults():

    dp_params={ 'max_grad_norm': 1.0,
                'target_epsilon': 64,
                'noise_multiplier': None,
                'grad_sample_mode': "augmult",
                'transform_list': FAST_K1,
                }

    log_params={'save_model': None,
                'experiment_name': None,
                'logger': None,}

    training_params={
                    'num_labels': 2,
                    'epochs': 10,
                    "early_stop_patience": 5,
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
    logger = get_file_logger("sweep-clip", "K1_DP.log")
    hypparams.update({"logger":logger,})

    for gsn in [0.1,0.5,1,2,5,10,15,100]:
        name = f"test norm:{gsn},eps:64"
        lr=gsn/20*0.01 # TODO super wrong
        hypparams.update({"lr":lr,
                          "max_grad_norm":gsn,
                          "experiment_name": name})
        
        safe_sst2(**hypparams)
    

def non_dp_K1():
    hypparams=get_defaults()
    params = {
            'max_grad_norm': 10.0,
            'target_epsilon': 256,
            'dataset_size': 25000,
        }
    hypparams.update(params)

    logger = get_file_logger("sweep-lr", "K1_DP.log")
    hypparams.update({"logger":logger,})

    for lr in [0.001,0.01,0.05,0.1,0.5,1]:
        name = f"Halfsize LR sweep:{lr},test norm:{params['max_grad_norm']},eps:{params['target_epsilon']} (patient)"
        hypparams.update({"lr":lr,"experiment_name": name})
        safe_sst2(**hypparams)

    

def realistic_eps():
    #to see if performance comes close
    s = f"sst2"
    logger = get_file_logger(s, "Realistic.log")

    params = {
                'epochs': 5,
                'batch_size': 500,
                'lr': 0.05,
                'dataset_size': None,
                'max_grad_norm': 1.0,
                'target_epsilon': 8,
                #'noise_multiplier': 0.1,
                'save_model': "epsilon8.ckpt",
                'experiment_name': s,
                'logger': logger,
                'trainable_param_setter': classifier_and_pooler,
                'grad_sample_mode': "augmult",
                'transform_list': Augmentations().special_blend_no_reasoning(),

            }
    safe_sst2(**params)


def main():
    non_dp_K1()

if __name__ == "__main__":
    main()
