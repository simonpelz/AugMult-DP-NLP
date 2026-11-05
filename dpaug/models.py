"""Model loading + fine-tuning regimes (head / bitfit / lora / full) for encoders and decoders."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

from torch import nn
from transformers import AutoModelForSequenceClassification, PretrainedConfig

HEAD_KEYS = ("classifier", "score", "pooler")  # BERT/RoBERTa heads, decoder `score`, BERT pooler (thesis setup)


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_trainable(model: nn.Module, mode: str, head_keys: Iterable[str] = HEAD_KEYS) -> nn.Module:
    """In-place requires_grad for 'full' | 'head' | 'bitfit' (bias + head)."""
    for name, p in model.named_parameters():
        is_head = any(k in name for k in head_keys)
        if mode == "full":
            p.requires_grad = True
        elif mode == "head":
            p.requires_grad = is_head
        elif mode == "bitfit":
            p.requires_grad = is_head or name.endswith("bias")
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return model


def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0, target_modules=None) -> nn.Module:
    """Wrap with PEFT LoRA (task SEQ_CLS => classification head stays trainable via modules_to_save)."""
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules)
    return get_peft_model(model, cfg)


def load_seq_classifier(
    name_or_config: Union[str, PretrainedConfig],
    num_labels: int,
    mode: str = "full",
    model_cls=AutoModelForSequenceClassification,
    attn_implementation: Optional[str] = "eager",
    lora_kwargs: Optional[dict] = None,
    **from_pretrained_kwargs,
) -> nn.Module:
    """Load an encoder or decoder sequence classifier in a given fine-tuning regime.

    ``attn_implementation="eager"`` is the safe default under ``torch.func.vmap``.
    Decoder classifiers (Llama/Qwen/Gemma ``...ForSequenceClassification``) pool the
    last non-pad token, so ``pad_token_id`` must be set; we default it to eos/0.
    """
    if isinstance(name_or_config, PretrainedConfig):
        cfg = name_or_config
        cfg.num_labels = num_labels
        if attn_implementation:
            cfg._attn_implementation = attn_implementation
        model = model_cls.from_config(cfg)
    else:
        kw = dict(num_labels=num_labels, **from_pretrained_kwargs)
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        model = model_cls.from_pretrained(name_or_config, **kw)
    if getattr(model.config, "pad_token_id", None) is None:
        eos = getattr(model.config, "eos_token_id", None)
        model.config.pad_token_id = (eos[0] if isinstance(eos, (list, tuple)) else eos) if eos is not None else 0
    if mode == "lora":
        for p in model.parameters():
            p.requires_grad = False
        model = apply_lora(model, **(lora_kwargs or {}))
    else:
        set_trainable(model, mode)
    return model
