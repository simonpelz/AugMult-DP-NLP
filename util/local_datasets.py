
import os
from pathlib import Path
import tqdm

import csv
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import signal


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
                    'label': int(row['label']),

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

class ZHPrecomputedAugsDataset(AbstractDataset):
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
                    'label': int(row['label']),

                    'zh1': row['zh1'],
                    'zh2': row['zh2'],
                })
        return samples

    def _get_column_names(self):
        return ['sentence1', 'sentence2', 'label', 'zh1','zh2',]


#-------------------------------------------------------------------------------------
#--------------------------------precomputation---------------------------------------
#-------------------------------------------------------------------------------------


def get_last_completed_line(last_line_file):
    if os.path.exists(last_line_file):
        with open(last_line_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_last_completed_line(last_line_file,line_num):
    with open(last_line_file, 'w') as f:
        f.write(str(line_num))

def main(fromfile,tofile,progressfile):
    last_completed_line = get_last_completed_line(progressfile)
    start_line = last_completed_line + 1

    print("start loading translation models: ", end="")
    import nlpaug.augmenter.word as naw

    #bt_aug_zh = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-zh',to_model_name='Helsinki-NLP/opus-mt-zh-en',device="cpu",name="zh",).augment
    #print("first done | ", end="")
    bt_aug_de = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-de',to_model_name='Helsinki-NLP/opus-mt-de-en',device="cpu",name="de",).augment
    print("second done | ", end="")
    bt_aug_ru = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-ru',to_model_name='Helsinki-NLP/opus-mt-ru-en',device="cpu",name="ru",).augment
    print("third done | ", end="")
    bt_aug_ar = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-ar',to_model_name='Helsinki-NLP/opus-mt-ar-en',device="cpu",name="ar",).augment
    print("fourth done\nfinished loading models")

    with open(fromfile, 'r', encoding='utf8') as fin, open(tofile, 'a', encoding='utf8') as fout:
        
        def signal_handler(signum, frame):
            print("REQUEUEING - saving progress and exiting.")
            raise Exception

        signal.signal(signal.SIGTERM, signal_handler)

        reader = csv.DictReader(fin, delimiter='\t')
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')

        # Write header only if file is empty
        if os.path.getsize(tofile) == 0:
            writer.writerow(['label','sentence1', 'sentence2', 'zh1','zh2','de1','de2','ru1','ru2','ar1','ar2',])

        # Skip previously completed lines
        for _ in range(last_completed_line):
            next(reader)

        print("Writing to file...")
        for line_num, row in enumerate(tqdm.tqdm(reader), start=start_line):
            try:
                #zh1 = bt_aug_zh(row['sentence1'])[0]
                #zh2 = bt_aug_zh(row['sentence2'])[0]
                de1 = bt_aug_de(row['sentence1'])[0]
                de2 = bt_aug_de(row['sentence2'])[0]
                ru1 = bt_aug_ru(row['sentence1'])[0]
                ru2 = bt_aug_ru(row['sentence2'])[0]
                ar1 = bt_aug_ar(row['sentence1'])[0]
                ar2 = bt_aug_ar(row['sentence2'])[0]
                writer.writerow([row['label'],row['sentence1'],row['sentence2'],row['zh1'],row['zh2'],de1,de2,ru1,ru2,ar1,ar2,])
                
                # Flush and sync to ensure the line is committed to disk
                fout.flush()
                os.fsync(fout.fileno())
                # Only save progress after successful processing
                # errors are caused if preempted after flush and before progress update
                save_last_completed_line(progressfile,line_num)

            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                break  # Handle error, can retry on next run

                
        """               
        # index	genre	filename	year	old_index	source1	source2	sentence1	sentence2	score
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
    print("start resolving path")
    src_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for src_name, dst_name in zip(['train_precomputed_zh.tsv',], #'dev.tsv', 'test.tsv'],
                                  ['train_precomputed_5.tsv',]): #'dev.tsv', 'test.tsv']):
        source = src_dir / src_name
        dest = output_dir / dst_name
        progressfile = output_dir / 'last_processed_line_5.txt'
        main(source, dest,progressfile)

if __name__ == "__main__":
    create_biosses("/vol/aimspace/projects/physionet/mednli/processed","/vol/aimspace/projects/physionet/mednli/processed")