"""Parquet schema of the view store.

One row per (example, view). ``view_id == 0`` is always the unmodified original
(``method == "identity"``). Score columns are dynamic and always prefixed ``score_``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc

SCHEMA_VERSION = "1.0"
SCORE_PREFIX = "score_"
IDENTITY = "identity"

CORE_FIELDS = [
    pa.field("example_id", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("view_id", pa.int32(), nullable=False),
    pa.field("method", pa.string(), nullable=False),
    pa.field("generator_id", pa.string(), nullable=False),
    pa.field("text_a", pa.string(), nullable=False),
    pa.field("text_b", pa.string(), nullable=True),
    # Which text fields were changed w.r.t. the original, e.g. "a", "b", "ab", "" (identity).
    pa.field("augmented_fields", pa.string(), nullable=False),
    pa.field("label", pa.int64(), nullable=True),
    pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
]
# Extra columns written only by ``views.filter``.
FILTER_FIELDS = [
    pa.field("source_view_id", pa.int32(), nullable=False),
    pa.field("padded", pa.bool_(), nullable=False),
    pa.field("pad_source", pa.string(), nullable=True),  # None | "resample" | "identity"
]
CORE_NAMES = [f.name for f in CORE_FIELDS]


@dataclass
class ViewRecord:
    example_id: str
    split: str
    view_id: int
    method: str
    generator_id: str
    text_a: str
    text_b: Optional[str]
    augmented_fields: str
    label: Optional[int]
    created_at: datetime
    scores: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {n: getattr(self, n) for n in CORE_NAMES}
        for k, v in self.scores.items():
            row[k if k.startswith(SCORE_PREFIX) else SCORE_PREFIX + k] = v
        return row


def build_schema(score_columns: List[str], filtered: bool = False) -> pa.Schema:
    fields = list(CORE_FIELDS)
    if filtered:
        fields += FILTER_FIELDS
    fields += [pa.field(c, pa.float64(), nullable=True) for c in sorted(score_columns)]
    return pa.schema(fields, metadata={b"views_schema_version": SCHEMA_VERSION.encode()})


def score_columns_of(names) -> List[str]:
    return sorted(n for n in names if n.startswith(SCORE_PREFIX))


def validate_table(table: pa.Table, filtered: bool = False) -> None:
    """Raise ``ValueError`` if a table does not follow the store schema."""
    expected = build_schema(score_columns_of(table.column_names), filtered=filtered)
    missing = [f.name for f in expected if f.name not in table.column_names]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for f in expected:
        if table.schema.field(f.name).type != f.type:
            raise ValueError(f"column {f.name}: expected {f.type}, got {table.schema.field(f.name).type}")
    extra = [n for n in table.column_names if n not in expected.names]
    if extra:
        raise ValueError(f"unexpected columns: {extra}")
    view0 = table.filter(pc.equal(table["view_id"], 0))
    if view0.num_rows and set(view0["method"].to_pylist()) != {IDENTITY}:
        raise ValueError("view_id 0 must always be the identity view")
