from datasets import load_dataset
import torch
import nlpaug.augmenter.word as naw
from data import AbstractMultiViewTextDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optim
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine

MODEL_NAME = "bert-base-uncased"
EPOCHS = 2
BATCH_SIZE = 4
LR = 2e-5

def main():
    
    # Augmentations
    synonym_repl = naw.SynonymAug(aug_src='wordnet').augment
    context_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert").augment
    context_repl = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute").augment
    unaugmented = lambda x: x

    transform_list = [unaugmented, synonym_repl,context_insert,context_repl]

    # Load and preprocess the dataset
    pre_dataset = load_dataset("glue", "sst2")
    num_labels = pre_dataset["train"].features["label"].num_classes
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use only a subset of the training dataset for testing
    subset_train_dataset = pre_dataset["train"].select(range(20))
    mv_train_loader = AbstractMultiViewTextDataset(subset_train_dataset, tokenizer, transform_list=transform_list).__dataloader__(BATCH_SIZE)

    # Model configuration
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_labels
    print(f"num labels: {num_labels}")

    # Differential privacy parameters
    EPSILON = 8.0
    DELTA = 1 / len(mv_train_loader)
    MAX_GRAD_NORM = 1.0

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=mv_train_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0
    )

    # BatchMemoryManager for handling large batches safely
    with BatchMemoryManager(data_loader=dp_train_loader, max_physical_batch_size=32, optimizer=dp_optimizer) as memory_safe_data_loader: 
        for batch in memory_safe_data_loader:  # Iterating through augmented batches
            print(f"\n NEW BATCH"+"="*100+"\n")
            #print(f"stacked views batch: {batch}")
            for sample in batch['input_ids']:
                print(f"\n new sample: -------------------------------------\n")
                for view in sample:
                    # Convert tokens back to text and print them
                    example_to_string = tokenizer.decode(view, skip_special_tokens=True)
                    print(example_to_string)

if __name__ == "__main__":
    main()
