import argparse
import os
from pathlib import Path
import tqdm

import csv
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch


class AbstractDataset(Dataset, ABC):
    """Minimal list-of-dicts dataset mimicking the parts of the HF `datasets.Dataset` API used here."""

    def __init__(self):
        self.samples = self._load_data()

    @abstractmethod
    def _load_data(self):
        """Method to load data. Must be implemented by subclasses."""
        pass

    @property
    def column_names(self):
        # derived from the data so it stays correct after map/rename/remove
        return list(self.samples[0].keys()) if self.samples else []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample

    def select(self, indices):
        self.samples = [self.samples[i] for i in indices]
        return self

    def map(self, function, batched=False, batch_size=32):
        """Apply a function to each sample (the `batched` directive is ignored)."""
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
                sample.pop(name, None)
        return self


class LocalTSVDataset(AbstractDataset):
    """Sentence-pair TSV with columns sentence1, sentence2, label (MedNLI format)."""

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


def processHocLabel(label_str):
    result_label = []
    for i,index_value in enumerate(label_str.split(',')):
        [index,value] = index_value.split('_')
        assert i == int(index) # sanity check if dataset is correct
        result_label.append(int(value))
    return result_label


class LocalHoCDataset(AbstractDataset):
    """Hallmarks of Cancer (multi-label). Labels are multi-hot lists (10 classes in BLURB HoC)."""

    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        super().__init__()

    def _load_data(self):
        samples = []
        with open(self.tsv_file, 'r', encoding='utf8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                samples.append({
                    'sentence': row['sentence'],
                    'label': processHocLabel(row['labels']),
                })
        return samples


# ---------------------------------------------------------------------------------------------
# Precomputed views
#
# view name -> (TSV column of sentence1 view, TSV column of sentence2 view).
# The order of this dict is the canonical view order: "precomputed_A..." / K-scaling uses the
# first A views. It matches the column order of train_precomputed_16.tsv (original, the 11
# back-translation pivots, then 4 BlueBERT views), which is also the order in which the thesis
# files were built (train_precomputed_5 -> _12 -> _16).
#
# NOTE on the BlueBERT columns: the precomputation script (below) wrote headers
# bert_insert_zh_*, bert_replac_de_*, bert_replac_ru_*, but *all four* BlueBERT views were
# computed from the ORIGINAL sentences, not from the zh/de/ru back-translations. They are
# exposed here under honest names (two independent insert samples, two independent replace
# samples) while still reading the legacy headers.
# ---------------------------------------------------------------------------------------------
PRECOMPUTED_VIEW_COLUMNS = {
    "sentence": ("sentence1", "sentence2"),  # original (unaugmented)
    "zh": ("zh1", "zh2"),
    "de": ("de1", "de2"),
    "ru": ("ru1", "ru2"),
    "ar": ("ar1", "ar2"),
    "af": ("af1", "af2"),
    "fr": ("fr1", "fr2"),
    "es": ("es1", "es2"),
    "id": ("id1", "id2"),
    "it": ("it1", "it2"),
    "nl": ("nl1", "nl2"),
    "fi": ("fi1", "fi2"),
    "bert_insert_a": ("bert_insert_og_1", "bert_insert_og_2"),
    "bert_insert_b": ("bert_insert_zh_1", "bert_insert_zh_2"),   # legacy header is misleading, see note
    "bert_replace_a": ("bert_replac_de_1", "bert_replac_de_2"),  # legacy header is misleading, see note
    "bert_replace_b": ("bert_replac_ru_1", "bert_replac_ru_2"),  # legacy header is misleading, see note
}
CANONICAL_VIEW_ORDER = list(PRECOMPUTED_VIEW_COLUMNS)


class PrecomputedAugsDataset(AbstractDataset):
    """Sentence-pair TSV with precomputed augmented views.

    Each sample is {"label", "<view>1", "<view>2", ...} for the selected views, in the given order.
    views=None loads every known view present in the file (canonical order).
    """

    def __init__(self, tsv_file, views=None):
        self.tsv_file = tsv_file
        self.views = list(views) if views is not None else None
        super().__init__()

    def _load_data(self):
        samples = []
        with open(self.tsv_file, 'r', encoding='utf8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            header = set(reader.fieldnames or [])
            available = [v for v, cols in PRECOMPUTED_VIEW_COLUMNS.items() if all(c in header for c in cols)]
            if self.views is None:
                self.views = available
            unknown = [v for v in self.views if v not in PRECOMPUTED_VIEW_COLUMNS]
            missing = [v for v in self.views if v in PRECOMPUTED_VIEW_COLUMNS and v not in available]
            if unknown or missing:
                raise ValueError(f"Requested views not available in {self.tsv_file}: unknown={unknown}, "
                                 f"missing columns={missing}. Available: {available}")
            for row in reader:
                sample = {'label': int(row['label'])}
                for v in self.views:
                    c1, c2 = PRECOMPUTED_VIEW_COLUMNS[v]
                    sample[v + "1"] = row[c1]
                    sample[v + "2"] = row[c2]
                samples.append(sample)
        return samples



#-------------------------------------------------------------------------------------
#--------------------------------precomputation---------------------------------------
#-------------------------------------------------------------------------------------
# Offline script used to build train_precomputed_16.tsv from train_precomputed_12.tsv.
# It writes the LEGACY column names (see PRECOMPUTED_VIEW_COLUMNS note): all four BlueBERT
# columns are computed from the original sentences.

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

    """
    bt_aug_af = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-af',to_model_name='Helsinki-NLP/opus-mt-af-en',device="cuda",name="af",).augment
    bt_aug_fr = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-fr',to_model_name='Helsinki-NLP/opus-mt-fr-en',device="cuda",name="fr",).augment
    bt_aug_es = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-es',to_model_name='Helsinki-NLP/opus-mt-es-en',device="cuda",name="es",).augment
    bt_aug_id = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-id',to_model_name='Helsinki-NLP/opus-mt-id-en',device="cuda",name="id",).augment
    bt_aug_it = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-it',to_model_name='Helsinki-NLP/opus-mt-it-en',device="cuda",name="it",).augment
    bt_aug_nl = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-nl',to_model_name='Helsinki-NLP/opus-mt-nl-en',device="cuda",name="nl",).augment
    bt_aug_fi = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-fi',to_model_name='Helsinki-NLP/opus-mt-fi-en',device="cuda",name="fi",).augment
    """
    word_params = {"aug_p":0.1,"aug_max":15,}
    emb_param = {"top_k": 5,**word_params,"model_path":'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',"device": "cuda" if torch.cuda.is_available() else "cpu"}
    med_insert = naw.ContextualWordEmbsAug(**emb_param, action="insert",name="bert_insert").augment
    med_replac = naw.ContextualWordEmbsAug(**emb_param, action="substitute",name="bert_replace").augment
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
                             'fi1','fi2',
                             'bert_insert_og_1','bert_insert_og_2',
                             'bert_insert_zh_1','bert_insert_zh_2',
                             'bert_replac_de_1','bert_replac_de_2',
                             'bert_replac_ru_1','bert_replac_ru_2',
                             ])

        # Skip previously completed lines
        for _ in range(last_completed_line):
            next(reader)

        print("Writing to file...")
        for line_num, row in enumerate(tqdm.tqdm(reader), start=start_line):
            try:
                bert_insert_og_1 = med_insert(row['sentence1'])[0]
                bert_insert_og_2 = med_insert(row['sentence2'])[0]
                bert_insert_zh_1 = med_insert(row['sentence1'])[0]
                bert_insert_zh_2 = med_insert(row['sentence2'])[0]
                bert_replac_de_1 = med_replac(row['sentence1'])[0]
                bert_replac_de_2 = med_replac(row['sentence2'])[0]
                bert_replac_ru_1 = med_replac(row['sentence1'])[0]
                bert_replac_ru_2 = med_replac(row['sentence2'])[0]
                

                writer.writerow([row['label'],row['sentence1'],row['sentence2'],
                                 row['zh1'],
                                 row['zh2'],
                                 row['de1'],
                                 row['de2'],
                                 row['ru1'],
                                 row['ru2'],
                                 row['ar1'],
                                 row['ar2'],
                                 row['af1'],
                                 row['af2'],
                                 row['fr1'],
                                 row['fr2'],
                                 row['es1'],
                                 row['es2'],
                                 row['id1'],
                                 row['id2'],
                                 row['it1'],
                                 row['it2'],
                                 row['nl1'],
                                 row['nl2'],
                                 row['fi1'],
                                 row['fi2'],
                                 bert_insert_og_1,
                                 bert_insert_og_2,
                                 bert_insert_zh_1,
                                 bert_insert_zh_2,
                                 bert_replac_de_1,
                                 bert_replac_de_2,
                                 bert_replac_ru_1,
                                 bert_replac_ru_2,
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




def precompute(input_dir, output_dir, src_name='train_precomputed_12.tsv', dst_name='train_precomputed_16.tsv'):
    src_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = src_dir / src_name
    dest = output_dir / dst_name
    dest.touch(exist_ok=True)
    progressfile = output_dir / f'last_processed_line_{Path(dst_name).stem}.txt'
    main(source, dest, progressfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute BlueBERT views (resumable). "
                                                 "Defaults read/write $DATASET_DIR.")
    parser.add_argument("--input_dir", default=os.environ.get("DATASET_DIR"))
    parser.add_argument("--output_dir", default=os.environ.get("DATASET_DIR"))
    parser.add_argument("--src_name", default="train_precomputed_12.tsv")
    parser.add_argument("--dst_name", default="train_precomputed_16.tsv")
    args = parser.parse_args()
    if not args.input_dir or not args.output_dir:
        raise SystemExit("Pass --input_dir/--output_dir or set DATASET_DIR.")
    precompute(args.input_dir, args.output_dir, args.src_name, args.dst_name)
