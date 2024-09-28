from datasets import load_dataset
import torch
import nlpaug.augmenter.word as naw
from augmult.data import mv_dataloader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optim
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine

MODEL_NAME = "bert-base-uncased"
EPOCHS = 2
BATCH_SIZE = 10
LR = 2e-5
TASK = "qnli"

def main():
    
    # Augmentations
    synonym_repl = naw.SynonymAug(aug_src='wordnet').augment
    context_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert").augment
    context_repl = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute").augment
    unaugmented = lambda x: x

    transform_list = [unaugmented, synonym_repl,context_insert,context_repl]

    # Load and preprocess the dataset
    pre_dataset = load_dataset("glue", TASK)
    num_labels = pre_dataset["train"].features["label"].num_classes
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use only a subset of the training dataset for testing
    mv_train_loader = mv_dataloader(pre_dataset["train"],20,tokenizer,transform_list,BATCH_SIZE)
    
    def normal_dataloader(trainset, subset, tokenizer, transform_list, batch_size):
        if subset is not None:
            modified_trainset = trainset.select(range(subset))
        else:
            modified_trainset = trainset
            
        mv_train_loader = DataLoader(
            modified_trainset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4, # TODO optimize num workers dataloading
        )
        return mv_train_loader
    n_dl = normal_dataloader(pre_dataset["train"],20,tokenizer,transform_list,BATCH_SIZE)
    for batch in n_dl:
        print(batch)
    # Model configuration
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_labels

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

    reshape_flatten = lambda x: x.view(-1, x.size(-1))  

    K = len(transform_list)

    # BatchMemoryManager for handling large batches safely
    with BatchMemoryManager(data_loader=dp_train_loader, max_physical_batch_size=32, optimizer=dp_optimizer) as memory_safe_data_loader: 
        for batch in memory_safe_data_loader:  # Iterating through augmented batches
            print(f"\n NEW BATCH"+"="*100+"\n")

            print(f"stacked views batch: {batch}")

            # reshape batch flatten
            for column in batch:
                if column == 'labels':
                    # Duplicate labels from (N) to (N*K) relating to their samples and augmented versions
                    batch['labels'] = torch.repeat_interleave(batch['labels'], repeats=K, dim=0)                    
                else: 
                    batch[column] = reshape_flatten(batch[column])
            
            for k, v in batch.items():
                print(f"\ncolumn:\n{k},\n{v}\n---\n") 

            for sample in batch['input_ids']:
                print(f"\n new sample: -------------------------------------\n")
                # Convert tokens back to text and print them
                example_to_string = tokenizer.decode(sample, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                print(example_to_string)

if __name__ == "__main__":
    main()
