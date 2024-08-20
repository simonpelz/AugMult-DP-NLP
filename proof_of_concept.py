from datasets import load_dataset
import torch
import nlpaug.augmenter.word as naw
from data import MultiViewTextDataset
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
    mv_train_dataset = MultiViewTextDataset(subset_train_dataset, tokenizer, transform_list=transform_list)

    test_dataset = pre_dataset["validation"]
    tokens_test_dataset = test_dataset.map(
        lambda example: tokenizer(example["sentence"], max_length=128, padding='max_length', truncation=True),
        batched=True
    )

    mv_train_loader = DataLoader(mv_train_dataset, shuffle=False, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(tokens_test_dataset, shuffle=False, batch_size=BATCH_SIZE)

    # Model configuration
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Validate and fix the model for differential privacy if needed
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    model.train() #TODO does this belong somewhere else?

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
            for sample in batch[0]:
                print(f"\n new sample: -------------------------------------\n")
                for view in sample:
                    # Convert tokens back to text and print them
                    example_to_string = tokenizer.decode(view, skip_special_tokens=True)
                    print(example_to_string)

if __name__ == "__main__":
    main()
