import types
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch.optim as optim
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from train import train
from data import MultiViewTextDataset, prepare_eval_dataloaders
from privacy_engine_util import _prepare_model_modified
from augmentations import Augmentations

# Model Information
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2

# Training 
EPOCHS = 2
BATCH_SIZE = 4
LR = 5e-4

# Differential privacy parameters
EPSILON = 8.0
DELTA = 1 / 67349 # len(train_loader) #unaugmented samples
MAX_GRAD_NORM = 1.0


def model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_LABELS
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    return model, tokenizer


def main():
    
    # ----------- Initialisation -------------

    # Model
    model, tokenizer = model_and_tokenizer(MODEL_NAME, NUM_LABELS)
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # Augmentation types
    transform_list = Augmentations().test_augs()
    # Dataloaders
    pre_dataset = load_dataset("glue", "sst2")
    #TODO remove: Use only a subset of the training dataset for testing
    subset_train_dataset = pre_dataset["train"].select(range(20))
    mv_train_loader = MultiViewTextDataset(subset_train_dataset, tokenizer, transform_list=transform_list).__dataloader__(BATCH_SIZE)
    #valid_loader, test_loader = prepare_eval_dataloaders(pre_dataset, tokenizer, EVAL_BATCH_SIZE=64)

     # GPU handling
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)
    print(device_name)

    # ---------------- Make DP --------------------

    # Ensure DP compatibility
    model.train()
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()
    # A Hack to load the custom AugmultGradSamplerModule. Methods almost identical, 
    privacy_engine._prepare_model = types.MethodType(_prepare_model_modified, privacy_engine)

    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=mv_train_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        grad_sample_mode="augmult",
    )

    # Custom Grad Samplers
    # TODO is this handled by make_private? or implement


    # -------------- Training ----------------------
    
    train_inputs = {
        'dp_model': dp_model,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': len(transform_list),
        }
    
    for _ in range(EPOCHS):
        train(**train_inputs)

    # TODO add validation
    # TODO add logging
    # TODO add save ckpt


if __name__ == "__main__":
    main()
