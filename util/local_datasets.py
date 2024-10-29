
import os
from pathlib import Path
import tqdm

import csv
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch


class AbstractDataset(Dataset, ABC):
    def __init__(self):
        self.samples = self._load_data()
        self.column_names = self._get_column_names()

    @abstractmethod
    def _load_data(self):
        """Method to load data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_column_names(self):
        """Property to define column names. Must be implemented by subclasses."""
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
    
    def select(self, indices):
        self.samples = [self.samples[i] for i in indices]
        return self

    def map(self, function, batched=False, batch_size=32):
        """Apply a function to each sample or batch of samples in the dataset."""
        batched = False  # ignore batched directive
        if batched:
            new_samples = []
            for i in range(0, len(self.samples), batch_size):
                batch = self.samples[i:i+batch_size]
                new_batch = function(batch)
                new_samples.extend(new_batch)
            self.samples = new_samples
        else:
            for sample in self.samples:
                sample.update(function(sample))
        return self

    def rename_column(self, old_column_name, new_column_name):
        for sample in self.samples:
            if old_column_name in sample:
                sample[new_column_name] = sample.pop(old_column_name)
        return self

    def set_format(self, type=None, columns=None):
        if type == 'torch':
            for i, sample in enumerate(self.samples):
                for column in columns:
                    if column in sample:
                        self.samples[i][column] = torch.tensor(sample[column])
        else:
            raise NotImplementedError
        return self

    def remove_columns(self, column_names):
        for name in column_names:
            for sample in self.samples:
                if name in sample:
                    sample.pop(name)
                else:
                    break
        return self


class LocalTSVDataset(AbstractDataset):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        super().__init__()

    def _load_data(self):
        samples = []
        with open(self.tsv_file, 'r', encoding='utf8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                samples.append({
                    'sentence1': row['sentence1'],
                    'sentence2': row['sentence2'],
                    'label': int(row['label'])
                })
        return samples

    def _get_column_names(self):
        return ['sentence1', 'sentence2', 'label']

class PrecomputedAugsDataset(AbstractDataset):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        super().__init__()

    def _load_data(self):
        samples = []
        with open(self.tsv_file, 'r', encoding='utf8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                samples.append({
                    'sentence1': row['sentence1'],
                    'sentence2': row['sentence2'],
                    'label': int(row['label']), # TODO change int label

                    'zh1': row['zh1'],
                    'zh2': row['zh2'],
                    'de1': row['de1'],
                    'de2': row['de2'],
                    'ru1': row['ru1'],
                    'ru2': row['ru2'],
                    'ar1': row['ar1'],
                    'ar2': row['ar2'],
                })
        return samples

    def _get_column_names(self):
        return ['sentence1', 'sentence2', 'label', 'zh1','zh2','de1','de2','ru1','ru2','ar1','ar2',]


def convertBIOSSES(fromfile,tofile):
    import nlpaug.augmenter.word as naw

    bt_aug_zh = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-zh',to_model_name='Helsinki-NLP/opus-mt-zh-en',device="cuda",name="zh",).augment
    bt_aug_de = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-de',to_model_name='Helsinki-NLP/opus-mt-de-en',device="cuda",name="de",).augment
    bt_aug_ru = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-ru',to_model_name='Helsinki-NLP/opus-mt-ru-en',device="cuda",name="ru",).augment
    bt_aug_ar = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-ar',to_model_name='Helsinki-NLP/opus-mt-ar-en',device="cuda",name="ar",).augment

    with open(fromfile, 'r', encoding='utf8') as fin, open(tofile, 'w', encoding='utf8') as fout:
            reader = csv.DictReader(fin, delimiter='\t')
            writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
            writer.writerow(['label','sentence1', 'sentence2', 'zh1','zh2','de1','de2','ru1','ru2','ar1','ar2',])

            for row in tqdm.tqdm(reader):
                writer.writerow([row['label'],row['sentence1'],row['sentence2'] 
                                                                                ,bt_aug_zh(row['sentence1'])[0]
                                                                                ,bt_aug_zh(row['sentence2'])[0]
                                                                                ,bt_aug_de(row['sentence1'])[0]
                                                                                ,bt_aug_de(row['sentence2'])[0]
                                                                                ,bt_aug_ru(row['sentence1'])[0]
                                                                                ,bt_aug_ru(row['sentence2'])[0]
                                                                                ,bt_aug_ar(row['sentence1'])[0]
                                                                                ,bt_aug_ar(row['sentence2'])[0]
                                                                                ,])

"""               # index	genre	filename	year	old_index	source1	source2	sentence1	sentence2	score
                samples.append({
                    'sentence1': row['sentence1'],
                    'sentence2': row['sentence2'],
                    'score': float(row['score']),

                    'zh1': row['zh1'],
                    'zh2': row['zh2'],
                    'de1': row['de1'],
                    'de2': row['de2'],
                    'ru1': row['ru1'],
                    'ru2': row['ru2'],
                    'ar1': row['ar1'],
                    'ar2': row['ar2'],
                })"""


def create_biosses(input_dir, output_dir):
    src_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for src_name, dst_name in zip(['train.tsv', 'dev.tsv', 'test.tsv'],
                                  ['train.tsv', 'dev.tsv', 'test.tsv']):
        source = src_dir / src_name
        dest = output_dir / dst_name
        convertBIOSSES(source, dest)

if __name__ == "__main__":
    create_biosses("./datasets/BIOSSES","./datasets/BIOSSES/precomputed")