
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

class K5PrecomputedAugsDataset(AbstractDataset):
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
                    'af1': row['af1'],
                    'af2': row['af2'],
                    'fr1': row['fr1'],
                    'fr2': row['fr2'],
                    'es1': row['es1'],
                    'es2': row['es2'],
                    'id1': row['id1'],
                    'id2': row['id2'],
                    'it1': row['it1'],
                    'it2': row['it2'],
                    'nl1': row['nl1'],
                    'nl2': row['nl2'],
                    'fi1': row['fi1'],
                    'fi2': row['fi2'],
                })
        return samples

    def _get_column_names(self):
        return ['label','sentence1', 'sentence2', 'zh1','zh2','de1','de2','ru1','ru2','ar1','ar2',
                             'af1','af2',
                             'fr1','fr2',
                             'es1','es2',
                             'id1','id2',
                             'it1','it2',
                             'nl1','nl2',
                             'fi1','fi2',]



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

    print("start loading translation models: ")#, end="")
    import nlpaug.augmenter.word as naw

    bt_aug_af = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-af',to_model_name='Helsinki-NLP/opus-mt-af-en',device="cuda",name="af",).augment
    bt_aug_fr = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-fr',to_model_name='Helsinki-NLP/opus-mt-fr-en',device="cuda",name="fr",).augment
    bt_aug_es = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-es',to_model_name='Helsinki-NLP/opus-mt-es-en',device="cuda",name="es",).augment
    bt_aug_id = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-id',to_model_name='Helsinki-NLP/opus-mt-id-en',device="cuda",name="id",).augment
    bt_aug_it = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-it',to_model_name='Helsinki-NLP/opus-mt-it-en',device="cuda",name="it",).augment
    bt_aug_nl = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-nl',to_model_name='Helsinki-NLP/opus-mt-nl-en',device="cuda",name="nl",).augment
    bt_aug_fi = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-fi',to_model_name='Helsinki-NLP/opus-mt-fi-en',device="cuda",name="fi",).augment

    print("\nfinished loading models")

    with open(fromfile, 'r', encoding='utf8') as fin, open(tofile, 'a', encoding='utf8') as fout:
        
        reader = csv.DictReader(fin, delimiter='\t')
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')

        # Write header only if file is empty
        if os.path.getsize(tofile) == 0:
            writer.writerow(['label','sentence1', 'sentence2', 'zh1','zh2','de1','de2','ru1','ru2','ar1','ar2',
                             'af1','af2',
                             'fr1','fr2',
                             'es1','es2',
                             'id1','id2',
                             'it1','it2',
                             'nl1','nl2',
                             'fi1','fi2',])

        # Skip previously completed lines
        for _ in range(last_completed_line):
            next(reader)

        print("Writing to file...")
        for line_num, row in enumerate(tqdm.tqdm(reader), start=start_line):
            try:
                af1 = bt_aug_af(row['sentence1'])[0]
                af2 = bt_aug_af(row['sentence2'])[0]
                fr1 = bt_aug_fr(row['sentence1'])[0]
                fr2 = bt_aug_fr(row['sentence2'])[0]
                es1 = bt_aug_es(row['sentence1'])[0]
                es2 = bt_aug_es(row['sentence2'])[0]
                id1 = bt_aug_id(row['sentence1'])[0]
                id2 = bt_aug_id(row['sentence2'])[0]
                it1 = bt_aug_it(row['sentence1'])[0]
                it2 = bt_aug_it(row['sentence2'])[0]
                nl1 = bt_aug_nl(row['sentence1'])[0]
                nl2 = bt_aug_nl(row['sentence2'])[0]
                fi1 = bt_aug_fi(row['sentence1'])[0]
                fi2 = bt_aug_fi(row['sentence2'])[0]

                writer.writerow([row['label'],row['sentence1'],row['sentence2'],
                                 row['zh1'],
                                 row['zh2'],
                                 row['de1'],
                                 row['de2'],
                                 row['ru1'],
                                 row['ru2'],
                                 row['ar1'],
                                 row['ar2'],
                                af1,
                                af2,
                                fr1,
                                fr2,
                                es1,
                                es2,
                                id1,
                                id2,
                                it1,
                                it2,
                                nl1,
                                nl2,
                                fi1,
                                fi2,
                                 ])
                
                # Flush and sync to ensure the line is committed to disk
                fout.flush()
                os.fsync(fout.fileno())
                # Only save progress after successful processing
                # errors are caused if preempted after flush and before progress update
                save_last_completed_line(progressfile,line_num)

            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                break  # Handle error, can retry on next run


def precompute(input_dir, output_dir):
    print("start resolving path")
    src_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for src_name, dst_name in zip(['train_precomputed.tsv',], #'dev.tsv', 'test.tsv'],
                                  ['train_precomputed_12.tsv',]): #'dev.tsv', 'test.tsv']):
        source = src_dir / src_name
        dest = output_dir / dst_name
        progressfile = output_dir / 'last_processed_line_12.txt'
        main(source, dest,progressfile)

if __name__ == "__main__":
    precompute("/home/spelz/AugMult_DP_NLP/datasets/processed","/home/spelz/AugMult_DP_NLP/datasets/processed")