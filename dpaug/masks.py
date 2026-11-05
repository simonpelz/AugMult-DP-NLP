"""vmap-safe attention masks for HF transformers.

Transformers >= 5 builds attention masks with data-dependent shortcuts (e.g.
``padding_mask.all()``) that ``torch.func.vmap`` cannot trace. Passing an already
expanded **4D additive** mask takes the library's early-exit path, so we build it
outside vmap: ``(N, K, L)`` padding mask -> ``(N, K, 1, L, L)`` float mask with 0
where attention is allowed and ``finfo.min`` elsewhere (bidirectional for encoders,
causal & padding for decoders).

Limitation: models with per-layer mask variants (sliding-window layers in
Gemma-2/3, Mistral, Qwen2 with sliding window enabled) get the same full mask on
every layer — exact only while sequence length <= the window.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

ENCODER_MODEL_TYPES = {
    "bert",
    "roberta",
    "xlm-roberta",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "albert",
    "camembert",
    "mpnet",
    "modernbert",
    "neobert",
    "eurobert",
}


def is_causal_model(model) -> bool:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None)
    if mt is None:
        return False
    return mt not in ENCODER_MODEL_TYPES


def expand_padding_mask(mask: torch.Tensor, causal: bool, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(..., L)`` {0,1} padding mask -> ``(..., 1, L, L)`` additive mask."""
    keep = mask.to(torch.bool)[..., None, None, :]  # (..., 1, 1, L) keys
    length = mask.shape[-1]
    if causal:
        tri = torch.ones(length, length, dtype=torch.bool, device=mask.device).tril()
        keep = keep & tri  # (..., 1, L, L)
    else:
        keep = keep.expand(*mask.shape[:-1], 1, length, length)
    out = torch.zeros(keep.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~keep, torch.finfo(dtype).min)


def prepare_inputs(inputs: Mapping[str, torch.Tensor], model, mask_mode: Optional[str] = "auto"):
    """Return a copy of ``inputs`` whose ``(N, K, L)`` attention_mask is expanded to 4D-per-example."""
    if mask_mode is None or not isinstance(inputs, Mapping) or "attention_mask" not in inputs:
        return inputs
    am = inputs["attention_mask"]
    if am.dim() != 3:  # already expanded (N, K, 1, L, L) or something custom
        return inputs
    causal = is_causal_model(model) if mask_mode == "auto" else (mask_mode == "causal")
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    out = dict(inputs)
    out["attention_mask"] = expand_padding_mask(am, causal=causal, dtype=dtype)
    return out
