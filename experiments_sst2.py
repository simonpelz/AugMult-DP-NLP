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


def lr():

    s = f"lr"
    logger = get_file_logger(s, "lr.log")

    params = {
                'epochs': 5,
                'batch_size': 600,
                'lr': None,
                'dataset_size': 12000,
                'max_grad_norm': 1.0,
                'target_epsilon': 32,
                #'noise_multiplier': 0.5,
                #'save_model': None,
                'experiment_name': s,
                'logger': logger,
                'trainable_param_setter': classifier_and_pooler,
                'grad_sample_mode': "augmult",
                'transform_list': Augmentations().synonyms(2),
            }

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


def non_dp_K1():
    params = {
            'epochs': 10,
            'batch_size': 256,
            'lr': 0.01,
            #'dataset_size': 512,
            'max_grad_norm': 300.0,
            'noise_multiplier': 0.0,
            'save_model': "only bias non dp.ckpt",
            'experiment_name': "test_different_finetune_configurations"

        }

    s = f"Experiments group: datasetsize full, non-DP but my implementation, Only biases and classifier weights"
    logger_1 = get_file_logger(s, "augmult_K1_non_DP.log")
    logger_1.info("\n\n"+s)

    # experiment 1
    logger_1.info(json.dumps(params))
    params['trainable_param_setter'] = bias_and_classifier # json doesnt like functions
    params['grad_sample_mode'] = "bias_only"    
    params['transform_list'] = Augmentations().no_augmentations()
    safe_sst2(**params, logger=logger_1)
    

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
    realistic_eps()

if __name__ == "__main__":
    main()
