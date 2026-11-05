"""Dataset loading with column mapping into the canonical (example_id, text_a, text_b, label) form.

Supported ``dataset.type``:
  * ``hub``    -- ``datasets.load_dataset(path, name, split=..., revision=...)`` (public datasets)
  * ``local``  -- local csv/tsv/json/parquet files via ``datasets.load_dataset(<builder>, data_files=...)``.
                  Use this for credentialed data (e.g. MedNLI). Such data must stay local: never send it to
                  hosted APIs and never commit generated views (``outputs/`` is git-ignored).
  * ``inline`` -- ``rows: [{...}, ...]`` given directly in the config (tests / demos).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from views.utils import resolve_dataset_revision


@dataclass
class Example:
    example_id: str
    text_a: str
    text_b: Optional[str]
    label: Optional[int]


def load_examples(ds_cfg: Dict[str, Any], limit: Optional[int] = None) -> tuple[List[Example], dict]:
    """Return canonical examples plus a provenance dict for the manifest."""
    dtype = ds_cfg.get("type", "hub")
    split = ds_cfg.get("split", "train")
    cols = {"text_a": "text_a", "text_b": None, "label": "label", "id": None, **ds_cfg.get("columns", {})}
    prov: Dict[str, Any] = {"type": dtype, "split": split, "columns": cols}

    if dtype == "inline":
        rows = list(ds_cfg["rows"])
    else:
        import datasets

        if dtype == "hub":
            path, name, rev = ds_cfg["path"], ds_cfg.get("name"), ds_cfg.get("revision")
            ds = datasets.load_dataset(path, name, split=split, revision=rev)
            prov.update(path=path, name=name, revision=resolve_dataset_revision(path, rev))
        elif dtype == "local":
            builder = ds_cfg.get("format", "csv")
            kwargs = {"delimiter": "\t"} if builder == "tsv" else {}
            builder = "csv" if builder == "tsv" else builder
            files = ds_cfg["data_files"]
            files = [os.path.expandvars(f) for f in files] if isinstance(files, list) else os.path.expandvars(files)
            ds = datasets.load_dataset(builder, data_files={split: files}, split=split, **kwargs)
            prov.update(data_files=ds_cfg["data_files"], format=ds_cfg.get("format", "csv"))
            try:  # record a content fingerprint, not the data
                prov["fingerprint"] = ds._fingerprint
            except Exception:
                pass
        else:
            raise ValueError(f"unknown dataset.type: {dtype}")
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        rows = ds.to_list()

    if limit is not None:
        rows = rows[:limit]
    examples = []
    for i, r in enumerate(rows):
        label = r.get(cols["label"]) if cols["label"] else None
        examples.append(
            Example(
                example_id=str(r[cols["id"]]) if cols["id"] else str(i),
                text_a=str(r[cols["text_a"]]),
                text_b=str(r[cols["text_b"]]) if cols["text_b"] else None,
                label=None if label is None else int(label),
            )
        )
    prov["num_examples"] = len(examples)
    return examples, prov
