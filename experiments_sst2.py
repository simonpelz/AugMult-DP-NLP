import json
import torch
from augmult.augmentations import Augmentations
from util.logging_util import get_file_logger
from sst2 import sst2
from util.different_finetune_modes import *


def safe_sst2(logger,**kwargs):
    try:
        sst2(**kwargs,logger=logger)
    except torch.OutOfMemoryError as e:
        logger.error(e)


def augmentation_times():
    
    params = {
                'epochs': 2,
                'batch_size': 32,
                'lr': 0.05,
                'dataset_size': 1000,
                'max_grad_norm': 5.0,
                'noise_multiplier': 0.1,
            }

    s = f"Experiments group: Small datasetsize ({params['dataset_size']}), Augmentations vary"
    logger_2 = get_file_logger(s, "Different_augs.log")
    logger_2.info("\n\n"+s)
    
    a = Augmentations()
    con_rep = [a.unaugmented, a.context_replacement]
    con_ins = [a.unaugmented, a.context_insert]
    back_trans = [a.unaugmented, a.back_translate]
    typo = [a.unaugmented, a.typo]
    del_w = [a.unaugmented, a.del_word]
    swap_c = [a.unaugmented, a.swap_char]

    loop = [[a.unaugmented,a.synonym_ppdb]]


    for t_list in loop:
        logger_2.info(f"\n\n {t_list[1]}")
        torch.cuda.empty_cache() 
        safe_sst2(**params, logger=logger_2,transform_list=t_list)


def test_working():
    params = {
            'epochs': 2,
            'batch_size': 10,
            'lr': 0.02,
            'dataset_size': 100,
            'max_grad_norm': 2.0,
            'noise_multiplier': 1.0,
            #'save_model': "discard.ckpt"
            'experiment_name': "testing"
        }

    s = f"test if still working"
    logger_1 = get_file_logger(s, "scrap.log")
    logger_1.info("\n\n"+s)
    logger_1.info(json.dumps(params))
    params['trainable_param_setter'] = bias_and_classifier # json doesnt like functions
    params['grad_sample_mode'] = "bias_only"
    params['transform_list'] = Augmentations().synonyms(4)

    safe_sst2(**params, logger=logger_1)


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
    
def main():
    test_working()

if __name__ == "__main__":
    main()
