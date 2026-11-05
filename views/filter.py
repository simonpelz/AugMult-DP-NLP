"""Filter a view store by label-preservation scores and emit exactly K views per example.

    python -m views.filter --input outputs/views/<run> --k 8 --min-entail 0.8 [--min-sim 0.7] \
        [--pad resample|identity] [--select first|random] [--methods llm_paraphrase backtranslation]

K counts *all* views including the identity view (view 0), matching the thesis' transform lists
(``[unaugmented, aug_1, ...]``). If fewer than K-1 views pass, the example is padded:
``resample`` draws passing views again (seeded, with replacement) and falls back to identity
duplicates if none passed; ``identity`` pads with copies of the original. Padding is recorded
per row (``padded``, ``pad_source``) and summarised in the manifest -- note that padding with
identity copies up-weights the original inside the AugMult average.

Filtering is per example and uses only that example's scores, so it is compatible with AugMult's
DP-SGD guarantee; the thresholds themselves are hyperparameters (see views/scorers.py).
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from views.schema import IDENTITY
from views.store import read_store, rows_to_table, summarize, write_store
from views.utils import git_info, stable_hash, stable_seed

# CLI flag -> (score column, comparison)
THRESHOLDS = {
    "min_entail": ("score_entail", ">="),
    "min_sim": ("score_sim", ">="),
    "min_edit": ("score_edit", ">="),
    "max_edit": ("score_edit", "<="),
    "min_len_ratio": ("score_len_ratio", ">="),
    "max_len_ratio": ("score_len_ratio", "<="),
    "min_task_agree": ("score_task_agree", ">="),
    "min_task_label_prob": ("score_task_label_prob", ">="),
}


def passes(row: dict, thresholds: Dict[str, float], methods: Optional[List[str]], generators: Optional[List[str]]) -> bool:
    if methods and not any(row["method"] == m or row["method"].startswith(m + "/") for m in methods):
        return False
    if generators and row["generator_id"] not in generators:
        return False
    for flag, value in thresholds.items():
        col, op = THRESHOLDS[flag]
        v = row.get(col)
        if v is None:
            return False
        if (op == ">=" and v < value) or (op == "<=" and v > value):
            return False
    return True


def filter_rows(rows: List[dict], k: int, thresholds: Dict[str, float], pad: str = "resample",
                select: str = "first", seed: int = 0, methods=None, generators=None) -> tuple[List[dict], Dict[str, Any]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if pad not in ("resample", "identity") or select not in ("first", "random"):
        raise ValueError("pad must be resample|identity, select must be first|random")
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r["example_id"], []).append(r)

    out: List[dict] = []
    stats: Dict[str, Any] = {"examples": len(groups), "examples_padded": 0, "pad_rows": {"resample": 0, "identity": 0},
                             "passing_per_example_hist": {}, "pass_rate_by_method": {}}
    seen_by_method: Dict[str, List[int]] = {}
    for ex_id in sorted(groups, key=lambda x: (len(x), x)):
        g = sorted(groups[ex_id], key=lambda r: r["view_id"])
        ident = [r for r in g if r["method"] == IDENTITY]
        if len(ident) != 1:
            raise ValueError(f"example {ex_id}: expected exactly one identity view, got {len(ident)}")
        ident = ident[0]
        ok = []
        for r in g:
            if r["method"] == IDENTITY:
                continue
            p = passes(r, thresholds, methods, generators)
            seen_by_method.setdefault(r["method"], [0, 0])
            seen_by_method[r["method"]][0] += 1
            seen_by_method[r["method"]][1] += int(p)
            if p:
                ok.append(r)
        h = stats["passing_per_example_hist"]
        h[str(len(ok))] = h.get(str(len(ok)), 0) + 1

        rng = random.Random(stable_seed(seed, ex_id))
        chosen = list(ok)
        if select == "random":
            rng.shuffle(chosen)
        chosen = [(r, False, None) for r in chosen[: k - 1]]
        need = (k - 1) - len(chosen)
        if need > 0:
            stats["examples_padded"] += 1
            for _ in range(need):
                if pad == "resample" and ok:
                    chosen.append((rng.choice(ok), True, "resample"))
                else:
                    chosen.append((ident, True, "identity"))
                stats["pad_rows"][chosen[-1][2]] += 1

        for new_id, (r, padded, src) in enumerate([(ident, False, None)] + chosen):
            out.append({**r, "view_id": new_id, "source_view_id": r["view_id"], "padded": padded, "pad_source": src})
    stats["pass_rate_by_method"] = {m: {"seen": s, "passed": p, "rate": p / s if s else None}
                                    for m, (s, p) in sorted(seen_by_method.items())}
    return out, stats


def main(argv: Optional[List[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="directory written by views.generate")
    ap.add_argument("--output", default=None)
    ap.add_argument("--k", type=int, required=True, help="total views per example incl. identity")
    for flag in THRESHOLDS:
        ap.add_argument("--" + flag.replace("_", "-"), dest=flag, type=float, default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="keep only these methods (prefix match on '/')")
    ap.add_argument("--generators", nargs="*", default=None, help="keep only these generator_ids")
    ap.add_argument("--pad", choices=["resample", "identity"], default="resample")
    ap.add_argument("--select", choices=["first", "random"], default="first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    table, parent = read_store(args.input)
    if "padded" in table.column_names:
        sys.exit("input is already a filtered store; filter the original generate output instead")
    thresholds = {f: getattr(args, f) for f in THRESHOLDS if getattr(args, f) is not None}
    for f in thresholds:
        if THRESHOLDS[f][0] not in table.column_names:
            sys.exit(f"--{f.replace('_', '-')} needs column {THRESHOLDS[f][0]}, which this store lacks (add the scorer)")
    spec = {"k": args.k, "thresholds": thresholds, "methods": args.methods, "generators": args.generators,
            "pad": args.pad, "select": args.select, "seed": args.seed}
    out_dir = Path(args.output or f"{str(Path(args.input)).rstrip('/')}-filtered-{stable_hash(spec, 8)}")
    if out_dir.exists() and not args.overwrite:
        sys.exit(f"{out_dir} exists; pass --overwrite")

    rows, stats = filter_rows(table.to_pylist(), args.k, thresholds, args.pad, args.select, args.seed,
                              args.methods, args.generators)
    out_table = rows_to_table(rows, filtered=True)
    manifest = {
        "kind": "filtered_views",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": {"path": str(args.input), "run_name": parent.get("run_name"), "config_hash": parent.get("config_hash"),
                   "created_at": parent.get("created_at")},
        "filter": spec,
        "filter_stats": stats,
        "summary": summarize(out_table),
        # carried over so a filtered store is self-describing
        "dataset": parent.get("dataset"),
        "generators": parent.get("generators"),
        "scorers": parent.get("scorers"),
        "scorers_that_saw_labels": parent.get("scorers_that_saw_labels"),
        "code": git_info(),
        "privacy_notes": parent.get("privacy_notes"),
    }
    write_store(out_table, manifest, out_dir)
    print(f"wrote {out_table.num_rows} rows ({stats['examples']} examples x {args.k}); "
          f"{stats['examples_padded']} examples padded -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
