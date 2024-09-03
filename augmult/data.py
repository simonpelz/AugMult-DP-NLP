from turtle import st
import torch
import numpy as np
from torch.utils.data import DataLoader


def non_dp_tokenize_Dataloader(dataset,tokenizer,batch_size):
    tokens = dataset.map(lambda x: tokenizer(x['sentence'], max_length=128, padding='max_length', truncation=True), batched=True)
    tokens = tokens.remove_columns(['idx','sentence']).rename_column("label", "labels") 
    tokens.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(tokens, shuffle=False, batch_size=batch_size)
    return dataloader


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
        return DataLoader(self,batch_size,shuffle=False,pin_memory=True)
    

