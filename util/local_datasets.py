
import csv
from torch.utils.data import Dataset
import torch

class LocalTSVDataset(Dataset):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.samples = self._load_data()
        self.column_names = ['sentence1','sentence2','label']

    def _load_data(self):
        samples = []
        with open(self.tsv_file, 'r', encoding='utf8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                # Ensure row contains 'sentence1', 'sentence2', 'label'
                samples.append({
                    'sentence1': row['sentence1'],
                    'sentence2': row['sentence2'],
                    'label': int(row['label'])
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
    
    def select(self,indices):
        self.samples = [self.samples[i] for i in indices]
        return self

    def map(self, function, batched=False, batch_size=32):
        """Apply a function to each sample or batch of samples in the dataset. Batched map not supported"""
        
        # ignore batched directive
        batched = False
        if batched:
            # Process the dataset in batches
            new_samples = []
            for i in range(0, len(self.samples), batch_size):
                batch = self.samples[i:i+batch_size]
                new_batch = function(batch)
                new_samples.extend(new_batch)
            self.samples = new_samples
        else:
            # Apply the function to each individual sample
            for sample in self.samples:
                sample.update(function(sample))
        return self  # Returning self allows for method chaining

    def rename_column(self, old_column_name, new_column_name):
        """Rename a column (key in sample dict) from old_column_name to new_column_name."""
        for sample in self.samples:
            if old_column_name in sample:
                sample[new_column_name] = sample.pop(old_column_name)
        return self  # Returning self allows for method chaining


    def set_format(self, type=None, columns=None):
        """Only type == torch !! Set the format of the dataset and select specific columns for PyTorch conversion."""
        if type == 'torch':
            # Convert specified columns to PyTorch tensors
            for i, sample in enumerate(self.samples):
                for column in columns:
                    if column in sample:
                        self.samples[i][column] = torch.tensor(sample[column])
        else: raise NotImplementedError
        return self  # Returning self allows for method chaining

    def remove_columns(self, column_names):
        """Rename a column (key in sample dict) from old_column_name to new_column_name."""
        for name in column_names:
            for sample in self.samples:
                if name in sample:
                    sample.pop(name)
                else:
                    break
        return self  # Returning self allows for method chaining

