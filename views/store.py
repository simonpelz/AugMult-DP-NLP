"""Reading/writing the Parquet view store and its manifest, plus summary statistics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from views.schema import IDENTITY, SCHEMA_VERSION, build_schema, score_columns_of, validate_table

VIEWS_FILE = "views.parquet"
MANIFEST_FILE = "manifest.json"
FLIP_ENTAIL_THRESHOLD = 0.5


def rows_to_table(rows: List[dict], filtered: bool = False) -> pa.Table:
    score_cols = score_columns_of({k for r in rows for k in r})
    schema = build_schema(score_cols, filtered=filtered)
    for r in rows:
        for c in score_cols:
            r.setdefault(c, None)
    table = pa.Table.from_pylist(rows, schema=schema)
    validate_table(table, filtered=filtered)
    return table


def write_store(table: pa.Table, manifest: Dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / VIEWS_FILE)
    manifest = {"schema_version": SCHEMA_VERSION, **manifest}
    with open(out_dir / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False, default=str)
    return out_dir


def read_store(path: str | Path) -> tuple[pa.Table, Dict[str, Any]]:
    path = Path(path)
    table = pq.read_table(path / VIEWS_FILE)
    manifest = json.loads((path / MANIFEST_FILE).read_text()) if (path / MANIFEST_FILE).exists() else {}
    return table, manifest


def _quantiles(vals: List[float]) -> Dict[str, float]:
    vals = sorted(vals)
    n = len(vals)

    def q(p):
        return vals[min(n - 1, int(round(p * (n - 1))))]

    return {"n": n, "mean": sum(vals) / n, "p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}


def summarize(table: pa.Table, flip_threshold: float = FLIP_ENTAIL_THRESHOLD) -> Dict[str, Any]:
    """Counts, per-method score quantiles and a label-flip-rate estimate (non-identity views only).

    ``label_flip_rate_estimate`` uses ``score_task_agree`` (public task classifier disagreement)
    if available, otherwise the fraction of views with bidirectional entailment below
    ``flip_threshold``. Both are *proxies* of label noise ("inherent error"), not ground truth.
    """
    rows = table.to_pylist()
    score_cols = score_columns_of(table.column_names)
    by_method: Dict[str, List[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    out: Dict[str, Any] = {
        "counts": {
            "examples": len({r["example_id"] for r in rows}),
            "rows": len(rows),
            "by_method": {m: len(v) for m, v in sorted(by_method.items())},
            "by_generator": {},
        },
        "score_summary": {},
        "label_flip_rate_estimate": {},
    }
    for r in rows:
        g = out["counts"]["by_generator"]
        g[r["generator_id"]] = g.get(r["generator_id"], 0) + 1
    for m, rs in sorted(by_method.items()):
        if m == IDENTITY:
            continue
        out["score_summary"][m] = {
            c: _quantiles(vals) for c in score_cols if (vals := [r[c] for r in rs if r[c] is not None])
        }
        if "score_task_agree" in score_cols and any(r["score_task_agree"] is not None for r in rs):
            v = [r["score_task_agree"] for r in rs if r["score_task_agree"] is not None]
            est = {"value": 1 - sum(v) / len(v), "basis": "public task classifier disagreement", "n": len(v)}
        elif "score_entail" in score_cols and any(r["score_entail"] is not None for r in rs):
            v = [r["score_entail"] for r in rs if r["score_entail"] is not None]
            est = {"value": sum(x < flip_threshold for x in v) / len(v),
                   "basis": f"bidirectional NLI entailment < {flip_threshold}", "n": len(v)}
        else:
            est = {"value": None, "basis": "no label-preservation scorer configured", "n": 0}
        out["label_flip_rate_estimate"][m] = est
    return out


def views_per_example(table: pa.Table) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in table["example_id"].to_pylist():
        counts[e] = counts.get(e, 0) + 1
    return counts


def load_grouped(path: str | Path, k: Optional[int] = None) -> Dict[str, List[dict]]:
    """Convenience loader for training code: ``{example_id: [view rows sorted by view_id]}``.

    If ``k`` is given, asserts every example has exactly ``k`` views (use a filtered store).
    """
    table, _ = read_store(path)
    groups: Dict[str, List[dict]] = {}
    for r in table.to_pylist():
        groups.setdefault(r["example_id"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["view_id"])
        if k is not None and len(g) != k:
            raise ValueError(f"example {g[0]['example_id']} has {len(g)} views, expected {k}")
    return groups
