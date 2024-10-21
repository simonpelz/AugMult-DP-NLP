
import csv
from torch.utils.data import Dataset

class LocalTSVDataset(Dataset):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.samples = self._load_data()

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

