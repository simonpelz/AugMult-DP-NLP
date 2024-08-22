import types
from datasets import load_dataset
import torch
import nlpaug.augmenter.word as naw
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch.optim as optim
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from train import train
from data import MultiViewTextDataset, prepare_eval_dataloaders
from privacy_engine_util import _prepare_model_modified


MODEL_NAME = "bert-base-uncased"
EPOCHS = 2
BATCH_SIZE = 4
LR = 5e-4

def main():
    
    # Augmentations
    synonym_repl = naw.SynonymAug(aug_src='wordnet').augment
    context_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert").augment
    context_repl = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute").augment
    unaugmented = lambda x: x

    transform_list = [unaugmented, synonym_repl,context_insert,context_repl]

    # Load and preprocess the dataset
    pre_dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use only a subset of the training dataset for testing
    subset_train_dataset = pre_dataset["train"].select(range(20))
    mv_train_loader = MultiViewTextDataset(subset_train_dataset, tokenizer, transform_list=transform_list).__dataloader__(BATCH_SIZE)
    #valid_loader, test_loader = prepare_eval_dataloaders(pre_dataset, tokenizer, EVAL_BATCH_SIZE=64)

    # Model configuration
    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_labels = pre_dataset["train"].features["label"].num_classes
    config.num_labels = num_labels
    print(f"Number of labels: {num_labels}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.train()

    # Ensure DP compatibility
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    # GPU handling
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(device_name)
    model.to(device)

    # Differential privacy parameters
    EPSILON = 8.0
    DELTA = 1 / len(mv_train_loader)
    MAX_GRAD_NORM = 1.0

    optimizer = optim.SGD(model.parameters(), lr=LR)

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

    train_inputs = {
        'dp_model': dp_model,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': len(transform_list),
        }
    
    train(**train_inputs)

if __name__ == "__main__":
    main()
