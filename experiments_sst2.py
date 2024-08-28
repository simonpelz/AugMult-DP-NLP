from augmentations import Augmentations
from logging_util import get_file_logger
from sst2 import sst2

def exp_1():
    params = {
            'epochs': 2,
            'batch_size': 32,
            'lr': 0.01,
            'dataset_size': 30000,
            'max_grad_norm': 15.0,
            'noise_multiplier': 0.1,
            'transform_list': Augmentations().K_2()
        }

    s = "Experiments group: Medium datasetsize (30K), One Augmentation (Synonym)"
    logger_1 = get_file_logger(s, "medium_dataset_one_aug.log")
    logger_1.info("\n\n"+s)

    # experiment 1
    logger_1.info("\n\n huge max clipping: 15, noise: 0.1, biggest avg_phys_batchsize im daring: 32*2, 2 epochs")
    sst2(**params, logger=logger_1)

    # experiment 2
    params['max_grad_norm'] = 5
    logger_1.info("\n\n clipping norm set to 5, rest stays the same")
    sst2(**params, logger=logger_1)

    # experiment 3
    params['max_grad_norm'] = 1
    logger_1.info("\n\n clipping norm set to default of 1, rest stays the same")
    sst2(**params, logger=logger_1)


def exp_2():
    

    s = "Experiments group: Small datasetsize (10K), Augmentations vary"
    logger_2 = get_file_logger(s, "Different_augs.log")
    logger_2.info("\n\n"+s)

    params = {
            'epochs': 2,
            'batch_size': 32,
            'lr': 0.01,
            'dataset_size': 10000,
            'max_grad_norm': 15.0,
            'noise_multiplier': 0.1,
            'logger' :logger_2,
        }
    # experiment 1
    logger_2.info("\n\n simple and fast augs")
    sst2(**params, transform_list=Augmentations().fast_augs())

    # experiment 2
    logger_2.info("\n\n All augs(7)")
    sst2(**params, transform_list=Augmentations().all_augs())
    
    
def main():
    exp_1()
    exp_2()

if __name__ == "__main__":
    main()
