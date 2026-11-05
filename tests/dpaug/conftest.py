import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import BertConfig, LlamaConfig, Qwen2Config  # noqa: E402

from dpaug.models import load_seq_classifier  # noqa: E402

VOCAB = 100


def tiny_bert_config():
    return BertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )


def tiny_llama_config():
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
    )


def tiny_qwen2_config():
    return Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
    )


CONFIGS = {"bert": tiny_bert_config, "llama": tiny_llama_config, "qwen2": tiny_qwen2_config}


def make_model(arch: str, mode: str, num_labels: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    return load_seq_classifier(CONFIGS[arch](), num_labels, mode=mode)


def make_inputs(n: int, k: int, length: int = 10, n_pad: int = 3, seed: int = 0):
    """Right-padded random token views; pad id 0 (== decoder pad_token_id)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, VOCAB, (n, k, length), generator=g)
    am = torch.ones_like(ids)
    if n_pad:
        am[..., length - n_pad :] = 0
        ids[..., length - n_pad :] = 0
    return {"input_ids": ids, "attention_mask": am}


@pytest.fixture
def make():
    return make_model
