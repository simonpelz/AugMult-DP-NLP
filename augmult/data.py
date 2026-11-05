from collections import defaultdict
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from util.local_datasets import LocalHoCDataset, LocalTSVDataset, PrecomputedAugsDataset
from util.reproducibility import seed_worker

LOCAL_DATASETS = ("mednli", "hoc")
MODEL_INPUT_COLUMNS = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']


def get_dataset(dataset_name, glue=True, precomputed_augs=False, precomputed_views=None,
                hf_path=None, hf_config=None, text_columns=None, label_column="label", splits=None):
    """Returns a dict with 'train', 'validation' and (if available) 'test' splits.

    Three sources:
      * glue=True:          HF `glue/<dataset_name>` (test labels are hidden, i.e. -1).
      * hf_path is set:     any HF hub dataset. `text_columns` (1 or 2 column names) and
                            `label_column` are renamed to sentence / sentence1,sentence2 and label;
                            `splits` maps {"train","validation","test"} -> hub split names.
      * otherwise:          local TSVs in $DATASET_DIR (mednli, hoc).
    """
    if glue:
        return dict(load_dataset("glue", dataset_name))
    if hf_path is not None:
        return get_hf_dataset(hf_path, hf_config, text_columns, label_column, splits)

    data_dir = os.environ.get("DATASET_DIR")
    if data_dir is None: raise ValueError("DATASET_DIR environment variable is not set.")
    train_file = os.path.join(data_dir, 'train.tsv')
    val_file = os.path.join(data_dir, 'dev.tsv')
    test_file = os.path.join(data_dir, 'test.tsv')

    if dataset_name == "mednli":
        if precomputed_augs:
            train = PrecomputedAugsDataset(os.path.join(data_dir, 'train_precomputed_16.tsv'), views=precomputed_views)
        else:
            train = LocalTSVDataset(train_file)
        dataset = {
            'train':        train,
            'validation':   LocalTSVDataset(val_file),
            'test':         LocalTSVDataset(test_file),
        }
    elif dataset_name == "hoc":
        if precomputed_augs: raise NotImplementedError("precomputed augmentations only exist for mednli")
        dataset = {
            'train':        LocalHoCDataset(train_file),
            'validation':   LocalHoCDataset(val_file),
            'test':         LocalHoCDataset(test_file),
        }
    else:
        raise NotImplementedError(f"Unknown local dataset {dataset_name!r} (known: {LOCAL_DATASETS}). "
                                  "Use --glue for GLUE tasks or --hf_path for other HF hub datasets.")
    return dataset


def get_hf_dataset(hf_path, hf_config=None, text_columns=None, label_column="label", splits=None):
    if not text_columns or len(text_columns) not in (1, 2):
        raise ValueError("text_columns must name 1 (single sentence) or 2 (sentence pair) columns")
    splits = {"train": "train", "validation": "validation", "test": "test", **(splits or {})}
    raw = load_dataset(hf_path, hf_config)
    targets = ["sentence"] if len(text_columns) == 1 else ["sentence1", "sentence2"]
    dataset = {}
    for ours, theirs in splits.items():
        if theirs is None or theirs not in raw:
            if ours != "test":
                raise ValueError(f"split {theirs!r} (for {ours}) not in {hf_path}: {list(raw)}")
            continue
        ds = raw[theirs]
        keep = list(text_columns) + [label_column]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        for src, dst in zip(list(text_columns) + [label_column], targets + ["label"]):
            if src != dst:
                ds = ds.rename_column(src, dst)
        # drop unlabelled examples (e.g. SNLI uses -1 for no gold label; hidden test sets become empty)
        ds = ds.filter(lambda ex: not (isinstance(ex["label"], int) and ex["label"] < 0))
        dataset[ours] = ds
    return dataset


def has_labels(split):
    """False for splits with hidden labels (GLUE test sets use -1)."""
    if split is None or len(split) == 0:
        return False
    label = split[0]["label"]
    return not (isinstance(label, int) and label < 0)


def tokenize_from_dataset(sample,tokenizer):
    # Probably very inefficient computation bc in .map, but is done once with small datasets
    if any(c in sample for c in ["premise","question","sentence2","question2"]):
        sentence_columns = [c for c in sample.keys() if c not in ('label', 'idx')]
        tokens = tokenizer(sample[sentence_columns[0]],sample[sentence_columns[1]], max_length=128, padding='max_length', truncation=True)
    elif "sentence" in sample:
        if (len([c for c in sample.keys() if c != 'idx'])-1!=1): raise NotImplementedError # might get false alarms for non glue but better safe than sorry
        tokens = tokenizer(sample['sentence'], max_length=128, padding='max_length', truncation=True)
    else:
        raise NotImplementedError
    return tokens


def non_dp_tokenize_dataloader(dataset,tokenizer,batch_size, shuffle=False, generator=None):
    if 'idx' in dataset.column_names: dataset = dataset.remove_columns(['idx'])
    tokens = dataset.map(lambda x: tokenize_from_dataset(x,tokenizer), batched=True)
    tokens = tokens.rename_column("label", "labels")
    # drop raw text columns (needed for the local list-of-dict datasets; HF would just hide them)
    tokens = tokens.remove_columns([c for c in tokens.column_names if c not in MODEL_INPUT_COLUMNS])
    tokens.set_format(type='torch', columns=[c for c in MODEL_INPUT_COLUMNS if c in tokens.column_names])
    dataloader = DataLoader(tokens, shuffle=shuffle, batch_size=batch_size, generator=generator)
    return dataloader


def unpack_dict_list(list_of_dicts):
    # unpack a list of dicts with same keys into a single dict with a list of each of the values
    combined_dict = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            combined_dict[key].append(value)
    combined_dict = dict(combined_dict)
    return combined_dict


def _labels_tensor(labels):
    # multi-label datasets (HoC) carry multi-hot lists -> float targets for BCEWithLogits
    if len(labels) > 0 and isinstance(labels[0], (list, tuple)):
        return torch.tensor(labels, dtype=torch.float)
    return torch.tensor(labels)


def init_mv_collate(tokenizer, transform_list,max_length=128, precomputed=False):
    if precomputed:
        return lambda batch: collate_precomputed(batch,tokenizer, transform_list,max_length)
    else:
        return lambda batch: collate_fn(batch,tokenizer, transform_list,max_length)


def _stack_views(grouped_by_aug):
    # convert to (token, mask etc.) shape to: [K, B, seq_len]
    # result is dict with {"input_ids":[[B,seq_len] x K ...],"atmask":[[B,seq_len] x K ...], etc.}
    # we want: {"input_ids":[B, K, seq_len],etc.} (test example: (11,2,128))
    combined_aug_groups = unpack_dict_list(grouped_by_aug)
    return {key: torch.stack(combined_aug_groups[key], dim=1) for key in grouped_by_aug[0].keys()}


def collate_precomputed(batch,tokenizer, transform_list,max_length=128):
    """
    assumes that there is only labels and view columns of the form <view>1 and <view>2
    (see util/local_datasets.PrecomputedAugsDataset; "sentence" is the original pair)
    """

    # transformlist will contain augs and then pad the rest with n_augs*n_precomputed
    # eg. for *one* precomputed backtranslate: [unaugmented,synonym,swap, None,None,None] (n=3 "None"s for every precomp.)

    # eliminate Nones
    multiply_by_following_augs = [aug for aug in transform_list if aug]
    assert len(multiply_by_following_augs) > 0 # at least put [unaugmented,...None]

    #bundle separate dicts into one
    combined_batch = unpack_dict_list(batch)
    tokenize_args = {'padding': 'max_length','truncation': True,
        'max_length': max_length,'return_tensors': 'pt'
    }
    labels = combined_batch.pop("label") # may raise error but this is only for training!

    # precomputed view names in dataset column order (deterministic; previously a `set`)
    precomp_augs = list(dict.fromkeys(s[:-1] for s in combined_batch.keys()))
    assert len(transform_list) == (len(multiply_by_following_augs)*len(precomp_augs)), \
        f"K={len(transform_list)} but data has {len(precomp_augs)} views x {len(multiply_by_following_augs)} augs"
    # load precomputed
    base_aug_sentence_pairs=[]
    for aug in precomp_augs:
        s1,s2 = combined_batch[(aug+"1")],combined_batch[(aug+"2")]
        base_aug_sentence_pairs.append([s1,s2])

    # augment and group by aug
    grouped_by_aug = []
    for multiply_aug in multiply_by_following_augs:
        # transform all precomputed
        for [s1,s2] in base_aug_sentence_pairs:
            #augment
            aug_s1= multiply_aug(list(s1))
            aug_s2= multiply_aug(list(s2))

            #tokenize
            tokenized_pairs = tokenizer(aug_s1,aug_s2,**tokenize_args) # is a dict from tokenizer
            # group all aug combinations separately
            grouped_by_aug.append(tokenized_pairs)

    stacked_views = _stack_views(grouped_by_aug)
    stacked_views['labels'] = _labels_tensor(labels)
    return stacked_views


def collate_fn(batch,tokenizer, transform_list,max_length=128):

    combined_batch = unpack_dict_list(batch)

    tokenize_args = {'padding': 'max_length','truncation': True,
        'max_length': max_length,'return_tensors': 'pt'
    }

    # relies on column names: everything except label/idx is treated as a text column
    labels = combined_batch.pop("label") # may raise error but this is only for training!
    _ = combined_batch.pop("idx", None) # some are called index or id

    grouped_by_aug = [[] for _ in range(len(transform_list))]
    for i, transform in enumerate(transform_list):

        # augment all sentences
        augmented_sentences = []
        for sentence in combined_batch.values():
            augmented_sentences.append(transform(list(sentence)))

        # tokenize based on sentence count
        if len(augmented_sentences)==1:
            tokenized_pairs = tokenizer(augmented_sentences[0],**tokenize_args)
        elif len(augmented_sentences)==2:
            tokenized_pairs = tokenizer(augmented_sentences[0],augmented_sentences[1],**tokenize_args)
        else: raise NotImplementedError
        grouped_by_aug[i] = tokenized_pairs # is a dict from tokenizer

    stacked_views = _stack_views(grouped_by_aug)
    stacked_views['labels'] = _labels_tensor(labels)
    return stacked_views



def dp_dataloader(trainset, subset, tokenizer, transform_list, batch_size, generator=None):
    if subset is not None:
        modified_trainset = trainset.select(range(subset))
    else:
        modified_trainset = trainset

    mv_train_loader = DataLoader(
        modified_trainset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
        worker_init_fn=seed_worker,
        # Opacus' DPDataLoader reuses this generator for Poisson sampling
        generator=generator,

        # collate will later be overwritten for empty batch handling!
        collate_fn=lambda batch: collate_fn(batch, tokenizer, transform_list)
    )
    return mv_train_loader
