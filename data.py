from turtle import st
import torch
import numpy as np
from torch.utils.data import DataLoader


def prepare_eval_dataloaders(pre_dataset,tokenizer, EVAL_BATCH_SIZE=64):
    """
    initialize dataloaders for validation and test dataset

    Returns:
        validation_dataloader, test_dataloader
    """
    validation_dataset = pre_dataset["validation"]
    tokens_validation_dataset = validation_dataset.map(
        lambda example: tokenizer(example["sentence"], max_length=128, padding='max_length', truncation=True),
        batched=True
    )
    test_dataset = pre_dataset["test"]
    tokens_test_dataset = test_dataset.map(
        lambda example: tokenizer(example["sentence"], max_length=128, padding='max_length', truncation=True),
        batched=True
    )

    validation_dataloader = DataLoader(tokens_validation_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE)
    test_dataloader = DataLoader(tokens_test_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE) 

    return validation_dataloader, test_dataloader


# TODO create check that len(transform_list) == K
# TODO is non augmented sample in K or K+1?
class MultiViewTextDataset(torch.utils.data.Dataset):
    """
    Extends the Pytorch Dataset Class to Augment text samples during Runtime and stacks them per Sample.
    """
    def __init__(self, pre_dataset, tokenizer, transform_list=None, max_length=128):
        self.pre_dataset = pre_dataset
        self.tokenizer = tokenizer
        self.transform_list = transform_list if transform_list else [lambda x: x]
        self.max_length = max_length
        self.key_to_text = 'sentence'
        self.key_to_labels = 'label'

    def __len__(self):
        # TODO what exactly is its len? k*len or len
        return len(self.pre_dataset)
    
    def __tokenize(self, text):
        tokenized_text = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        return tokenized_text
    
    def __augment_tokenize_stack(self, text):
        views = []
        for transform in self.transform_list:
            augmented_text = transform(text)
            tokenized_text = self.__tokenize(augmented_text)
            views.append(tokenized_text)
        
        # Stack views into a single tensor of shape [K, max_seq_len]
        stacked_views = {key: torch.stack([view[key].squeeze(0) for view in views]) for key in views[0].keys()}
        return stacked_views

    def __getitem__(self, index):
        text = self.pre_dataset[index][self.key_to_text]
        label = self.pre_dataset[index][self.key_to_labels]
        
        stacked_views = self.__augment_tokenize_stack(text)
        stacked_views['labels'] = label

        return stacked_views
    
    def __dataloader__(self, batch_size):
        return DataLoader(self,batch_size,shuffle=False)
    

