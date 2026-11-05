"""Shared fixtures: a tiny, offline BERT classifier and multi-view batches."""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

VOCAB = 100
SEQ_LEN = 8
NUM_LABELS = 3


def tiny_bert(seed: int = 0, trainable: str = "classifier_and_pooler"):
    """Tiny BertForSequenceClassification without network access.

    Dropout is disabled so that forward passes are deterministic (the GSM hooks
    only fire in train mode, so we cannot use eval() to get determinism).
    """
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(seed)
    cfg = BertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=37,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=NUM_LABELS,
    )
    model = BertForSequenceClassification(cfg)
    model.train()
    set_trainable(model, trainable)
    return model


def set_trainable(model, trainable: str):
    for p in model.parameters():
        p.requires_grad = False
    if trainable == "classifier_and_pooler":
        mods = [model.bert.pooler, model.classifier]
    elif trainable == "last_output_block":
        # Linear + LayerNorm that see token-level activations of shape (N*K, T, H)
        mods = [model.bert.encoder.layer[-1].output, model.bert.pooler, model.classifier]
    else:
        raise ValueError(trainable)
    for m in mods:
        for p in m.parameters():
            p.requires_grad = True


def multiview_batch(n: int, k: int, seed: int = 1):
    """Batch with shape [N, K, T] plus labels [N]; padding varies per view."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(5, VOCAB, (n, k, SEQ_LEN), generator=g)
    attention_mask = torch.ones(n, k, SEQ_LEN, dtype=torch.long)
    lengths = torch.randint(3, SEQ_LEN + 1, (n, k), generator=g)
    for i in range(n):
        for j in range(k):
            attention_mask[i, j, lengths[i, j]:] = 0
            input_ids[i, j, lengths[i, j]:] = 0
    token_type_ids = torch.zeros(n, k, SEQ_LEN, dtype=torch.long)
    labels = torch.randint(0, NUM_LABELS, (n,), generator=g)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def flatten_views(batch, k: int):
    """Same flattening as train.py: [N, K, T] -> [N*K, T], labels repeated K times."""
    out = {}
    for key, v in batch.items():
        if key == "labels":
            out[key] = torch.repeat_interleave(v, repeats=k, dim=0)
        else:
            out[key] = v.reshape(-1, v.size(-1))
    return out


@pytest.fixture
def tiny_model_factory():
    return tiny_bert
