"""Generate a view store from a YAML config.

    python -m views.generate --config configs/views/sst2_qwen.yaml --limit 100

Output: ``<output-dir>/<name>-<confighash>-<limit|all>/{views.parquet, manifest.json}``.
Everything runs locally; no hosted LLM APIs are used.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from views.config import config_hash, load_config, resolve_device
from views.data import Example, load_examples
from views.generators import Generator, build_generator
from views.schema import IDENTITY, ViewRecord
from views.scorers import build_scorer
from views.store import rows_to_table, summarize, write_store
from views.utils import env_versions, git_info, is_git_ignored

PRIVACY_NOTES = [
    "Views are computed per example from that example's own text only (generators never see labels or the other side of a pair).",
    "Scorers use public models; per-scorer 'saw_label' records whether the example's own label was used.",
    "Generated views may contain private text: keep outputs local and out of version control.",
    "Model-based views partly transfer the generator's distribution into the trained model (distillation confound); "
    "use generator_id / method columns to ablate it.",
]


def combine_pair_views(ex: Example, pools: Dict[str, List[tuple]], k: int, mode: str) -> List[dict]:
    """Turn per-field candidate pools into at most ``k`` views of the (possibly paired) example.

    ``alternate``: each view rewrites exactly one field (round-robin; falls back to the other field
    when one pool is exhausted). ``zip``: view j rewrites every field with its j-th candidate.
    """
    fields = [f for f in ("a", "b") if f in pools]
    if mode == "zip" and len(fields) == 2:
        n = min(k, len(pools["a"]), len(pools["b"]))
        return [{"text_a": pools["a"][j][0], "text_b": pools["b"][j][0], "augmented_fields": "ab",
                 "method": pools["a"][j][1] if pools["a"][j][1] == pools["b"][j][1] else f"{pools['a'][j][1]}+{pools['b'][j][1]}"}
                for j in range(n)]
    if mode not in ("alternate", "zip"):
        raise ValueError(f"unknown pair_combine mode {mode!r}")
    ptr = {f: 0 for f in fields}
    views = []
    j = 0
    while len(views) < k and any(ptr[f] < len(pools[f]) for f in fields):
        order = fields[j % len(fields):] + fields[:j % len(fields)]
        j += 1
        for f in order:
            if ptr[f] < len(pools[f]):
                text, tag = pools[f][ptr[f]]
                ptr[f] += 1
                views.append({"text_a": text if f == "a" else ex.text_a,
                              "text_b": text if f == "b" else ex.text_b,
                              "augmented_fields": f, "method": tag})
                break
    return views


def run_generator(gen: Generator, examples: List[Example], k: int, fields: List[str], mode: str) -> List[List[dict]]:
    pair = examples[0].text_b is not None if examples else False
    fields = [f for f in fields if f == "a" or pair]
    per_field_k = k if (len(fields) == 1 or mode == "zip") else math.ceil(k / len(fields)) + 1
    ids = [e.example_id for e in examples]
    pools: Dict[str, List[List[tuple]]] = {}
    for f in fields:
        texts = [e.text_a if f == "a" else e.text_b for e in examples]
        pools[f] = gen.generate(texts, ids, per_field_k, field=f)
    return [combine_pair_views(ex, {f: pools[f][i] for f in fields}, k, mode) for i, ex in enumerate(examples)]


def main(argv: Optional[List[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples")
    ap.add_argument("--output-dir", default="outputs/views")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-score", action="store_true", help="skip scorers")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg["device"])
    created_at = datetime.now(timezone.utc)
    run_name = f"{cfg['name']}-{config_hash(cfg)}-{args.limit if args.limit is not None else 'all'}"
    out_dir = Path(args.output_dir) / run_name
    if out_dir.exists() and not args.overwrite:
        sys.exit(f"{out_dir} exists; pass --overwrite")
    if is_git_ignored(out_dir) is False:  # None (outside any repo) is fine
        warnings.warn(f"{out_dir} is NOT git-ignored -- generated views may contain private text")

    t0 = time.time()
    examples, ds_prov = load_examples(cfg["dataset"], limit=args.limit)
    timing: Dict[str, Any] = {"load_data_s": round(time.time() - t0, 2)}
    split = cfg["dataset"].get("split", "train")
    k = int(cfg["k"])

    rows_per_ex: Dict[str, List[dict]] = {e.example_id: [] for e in examples}
    gen_provs, shortfall = [], {}
    for gcfg in cfg["generators"]:
        gen = build_generator(gcfg, seed=cfg["seed"], device=device)
        gk = int(gcfg.get("k", k))
        fields = list(gcfg.get("fields", ["a", "b"]))
        mode = gcfg.get("pair_combine", cfg["pair_combine"])
        t = time.time()
        all_views = run_generator(gen, examples, gk, fields, mode)
        dt = time.time() - t
        prov = gen.provenance()
        prov.update(k_requested=gk, fields=fields, pair_combine=mode)
        gen_provs.append(prov)
        n_views = sum(len(v) for v in all_views)
        timing[gen.generator_id] = {"seconds": round(dt, 2), "views": n_views,
                                    "examples_per_s": round(len(examples) / dt, 3) if dt > 0 else None}
        shortfall[gen.generator_id] = sum(gk - len(v) for v in all_views)
        for ex, views in zip(examples, all_views):
            for v in views:
                rows_per_ex[ex.example_id].append({"generator_id": gen.generator_id, **v})

    identity_gid = "identity"
    records: List[ViewRecord] = []
    by_id = {e.example_id: e for e in examples}
    for ex in examples:
        records.append(ViewRecord(ex.example_id, split, 0, IDENTITY, identity_gid, ex.text_a, ex.text_b, "",
                                  ex.label, created_at))
        for j, v in enumerate(rows_per_ex[ex.example_id], start=1):
            records.append(ViewRecord(ex.example_id, split, j, v["method"], v["generator_id"], v["text_a"],
                                      v["text_b"], v["augmented_fields"], ex.label, created_at))

    scorer_provs = []
    if not args.no_score:
        targets = [r for r in records if r.method != IDENTITY]
        origs = [by_id[r.example_id] for r in targets]
        vdicts = [{"text_a": r.text_a, "text_b": r.text_b, "augmented_fields": r.augmented_fields} for r in targets]
        for scfg in cfg["scorers"]:
            sc = build_scorer(scfg, device=device)
            t = time.time()
            for r, s in zip(targets, sc.score(origs, vdicts)):
                r.scores.update({f"score_{key}": val for key, val in s.items()})
            timing[f"scorer:{sc.name}"] = {"seconds": round(time.time() - t, 2), "views": len(targets)}
            scorer_provs.append(sc.provenance())

    table = rows_to_table([r.to_row() for r in records])
    manifest = {
        "kind": "views",
        "run_name": run_name,
        "created_at": created_at.isoformat(),
        "config_hash": config_hash(cfg),
        "config": {k_: v for k_, v in cfg.items() if not k_.startswith("_")},
        "config_path": cfg["_config_path"],
        "limit": args.limit,
        "device": device,
        "dataset": ds_prov,
        "generators": gen_provs,
        "scorers": scorer_provs,
        "scorers_that_saw_labels": [s["name"] for s in scorer_provs if s.get("saw_label")],
        "shortfall_views": shortfall,
        "timing": {**timing, "total_s": round(time.time() - t0, 2)},
        "summary": summarize(table),
        "code": git_info(),
        "env": env_versions(),
        "privacy_notes": PRIVACY_NOTES,
    }
    write_store(table, manifest, out_dir)
    print(f"wrote {table.num_rows} rows for {len(examples)} examples -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
