import torch
import numpy as np

# TODO create check that len(transform_list) == K
# TODO is non augmented sample in K or K+1?
class MultiViewTextDataset(torch.utils.data.Dataset):
    def __init__(self, pre_dataset, tokenizer, transform_list=None, max_length=128):
        self.pre_dataset = pre_dataset
        self.tokenizer = tokenizer
        self.transform_list = transform_list if transform_list else []
        self.max_length = max_length

    def __len__(self):
        # TODO what exactly is its len? k*len or len
        return len(self.pre_dataset)

    def __getitem__(self, index):
        text = self.pre_dataset[index]['sentence']
        label = self.pre_dataset[index]['label']
        
        views = []
        for transform in self.transform_list:
            augmented_text = transform(text)
            tokenized_input = self.tokenizer(
                augmented_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            views.append(tokenized_input)
        
        # Stack views into a single tensor of shape [K, max_seq_len]
        input_ids = torch.stack([view['input_ids'].squeeze(0) for view in views])
        attention_mask = torch.stack([view['attention_mask'].squeeze(0) for view in views])

        return input_ids, attention_mask, label
