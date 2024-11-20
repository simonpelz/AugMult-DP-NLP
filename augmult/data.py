from collections import defaultdict
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from util.local_datasets import  LocalHoCDataset, LocalTSVDataset, PrecomputedAugsDataset

def get_dataset(dataset_name,glue=True,precomputed_augs=False):
    if glue:
        dataset = load_dataset("glue", dataset_name)
    else:
        data_dir = os.environ.get("DATASET_DIR")
        if data_dir is None: raise ValueError("DATASET_DIR environment variable is not set.")
        train_file = os.path.join(data_dir, 'train.tsv')
        val_file = os.path.join(data_dir, 'dev.tsv')
        test_file = os.path.join(data_dir, 'test.tsv')
        if dataset_name == "mednli":
            if precomputed_augs:
                train_file = os.path.join(data_dir, 'train_precomputed_16.tsv')

                dataset = {
                    'train':        PrecomputedAugsDataset(train_file),
                    'validation':   LocalTSVDataset(val_file),
                    'test':         LocalTSVDataset(test_file)
                }

            else:
                dataset = {
                    'train':  LocalTSVDataset(train_file),
                    'validation':  LocalTSVDataset(val_file),
                    'test':  LocalTSVDataset(test_file)
                }
        if dataset_name == "hoc":
            dataset = {
                    'train':        LocalHoCDataset(train_file),
                    'validation':   LocalHoCDataset(val_file),
                    'test':         LocalHoCDataset(test_file)
                }
        else: raise NotImplementedError
    return dataset

def tokenize_from_dataset(sample,tokenizer):
    # Probably very inefficient computation bc in .map, but is done once with small datasets
    if any(c in sample for c in ["premise","question","sentence2","question2"]):
        sentence_columns = list(sample.keys())
        sentence_columns = [c for c in sentence_columns if c != 'label'] # .remove didnt work??
        tokens = tokenizer(sample[sentence_columns[0]],sample[sentence_columns[1]], max_length=128, padding='max_length', truncation=True)
    elif "sentence" in sample:
        if (len(sample.keys())-1!=1): raise NotImplementedError # might get false alarms for non glue but better safe than sorry
        tokens = tokenizer(sample['sentence'], max_length=128, padding='max_length', truncation=True)
    else:
        raise NotImplementedError
    return tokens


def non_dp_tokenize_dataloader(dataset,tokenizer,batch_size):
    if 'idx' in dataset.column_names: dataset = dataset.remove_columns(['idx'])
    tokens = dataset.map(lambda x: tokenize_from_dataset(x,tokenizer), batched=True)
    tokens = tokens.rename_column("label", "labels")
    for col in ["sentence1","sentence2","label","sentence"]: 
        if col in tokens.column_names: tokens.remove_columns([col])
    tokens.set_format(type='torch', columns=['input_ids', 'attention_mask','token_type_ids', 'labels'])
    dataloader = DataLoader(tokens, shuffle=False, batch_size=batch_size)
    return dataloader


def unpack_dict_list(list_of_dicts):  
    # unpack a list of dicts with same keys into a single dict with a list of each of the values  
    combined_dict = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            combined_dict[key].append(value)
    combined_dict = dict(combined_dict)
    return combined_dict


def init_mv_collate(tokenizer, transform_list,max_length=128, precomputed=False):
    if precomputed:
        return lambda batch: collate_precomputed(batch,tokenizer, transform_list,max_length)
    else:
        return lambda batch: collate_fn(batch,tokenizer, transform_list,max_length)


def collate_precomputed(batch,tokenizer, transform_list,max_length=128):
    """ 
    assumes that there is only labels, original sentences and augmented sentences as columns
    and string of form X1 and X2 where X is augmentation description for sentence1 and sentence2 (BERT)
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

    # get precomputed aug names
    precomp_augs = set([s[:-1] for s in combined_batch.keys()]) # of form sentence1 sentence2 zh1 zh2 etc
    assert len(transform_list) == (len(multiply_by_following_augs)*len(precomp_augs)) # K is defined by length of transformlist butis unused in this case. Pad list match K for fix
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
    
    # convert to (token, mask etc.) shape to: [K, B, seq_len]
    # result is dict with {"input_ids":[[B,seq_len] x K ...],"atmask":[[B,seq_len] x K ...], etc.}
    # we want: {"input_ids":[B, K, seq_len],etc.} (test example: (11,2,128))
    combined_aug_groups = unpack_dict_list(grouped_by_aug)
    
    stacked_views = defaultdict(list)
    # for each column from the tokenizer like "input_ids"
    for key in grouped_by_aug[0].keys():
        # group per sample (all augmented versions of that sample)
        for per_sample_aug in zip(*combined_aug_groups[key]):
            per_sample_aug = list(per_sample_aug)
            # and stack them into a new dict
            stacked_views[key].append(torch.stack(per_sample_aug))
    stacked_views = dict(stacked_views)
    stacked_views = {k: torch.stack(v) for k, v in stacked_views.items()}

    stacked_views['labels'] = torch.tensor(labels)

    return stacked_views
                        

def collate_fn(batch,tokenizer, transform_list,max_length=128):
    
    combined_batch = unpack_dict_list(batch)

    tokenize_args = {'padding': 'max_length','truncation': True,
        'max_length': max_length,'return_tensors': 'pt'
    }

    # TODO awful implementation relies on exact column names, but works for glue
    labels = combined_batch.pop("label") # may raise error but this is only for training!
    _ = combined_batch.pop("idx", None) # some are called index or id

    # TODO kinda hacky because column names are not the same
    # fails for additional columns that are not sentences
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

    # convert to (token, mask etc.) shape to: [K, B, seq_len]
    # result is dict with {"input_ids":[[B,seq_len] x K ...],"atmask":[[B,seq_len] x K ...], etc.}
    # we want: {"input_ids":[B, K, seq_len],etc.} (test example: (11,2,128))
    combined_aug_groups = unpack_dict_list(grouped_by_aug)

    stacked_views = defaultdict(list)
    # for each column from the tokenizer like "input_ids"
    for key in grouped_by_aug[0].keys():
        # group per sample (all augmented versions of that sample)
        for per_sample_aug in zip(*combined_aug_groups[key]):
            per_sample_aug = list(per_sample_aug)
            # and stack them into a new dict
            stacked_views[key].append(torch.stack(per_sample_aug))

    stacked_views = dict(stacked_views)
    stacked_views = {k: torch.stack(v) for k, v in stacked_views.items()}

    stacked_views['labels'] = torch.tensor(labels)

    return stacked_views



def dp_dataloader(trainset, subset, tokenizer, transform_list, batch_size):
    if subset is not None:
        modified_trainset = trainset.select(range(subset))
    else:
        modified_trainset = trainset

    #batches_per_epoch = -(len(modified_trainset)// -batch_size) # ceiling div
        
    mv_train_loader = DataLoader(
        modified_trainset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,

        # collate will later be overwritten for empty batch handling!
        collate_fn=lambda batch: collate_fn(batch, tokenizer, transform_list)
    )
    return mv_train_loader

"""
# Old code, too slow

def dp_dataloader(trainset,subset,tokenizer,transform_list,batch_size):
    if subset is not None:
        modified_trainset = trainset.select(range(subset))
    else:
        modified_trainset = trainset
    mv_train_set = MultiViewTextDataset(modified_trainset, tokenizer, transform_list=transform_list)
    mv_train_loader = DataLoader(mv_train_set,batch_size,shuffle=False,pin_memory=True,num_workers=0)
    return mv_train_loader


class MultiViewTextDataset(torch.utils.data.Dataset):

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
    
"""