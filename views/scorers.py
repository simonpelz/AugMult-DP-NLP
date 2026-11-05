"""Label-preservation / quality scorers for views.

Privacy contract
----------------
* Scorers use **public models only** (sentence-transformers, public NLI cross-encoders, public
  task classifiers). A "public" classifier must not have been trained on the private dataset.
* A scorer sees exactly one example at a time: the original text(s), the view text(s) and, only
  if ``saw_label`` is True, *that example's own label*. Never other examples or labels.
  Because AugMult computes everything per example *before* clipping, any filter/weight built
  from such per-example scores is covered by the DP-SGD guarantee for the downstream model.
* Caveat: filter *thresholds* are global hyperparameters. Choosing them by inspecting the
  score distribution of the private training set is a (usually unaccounted) data-dependent choice,
  like any hyperparameter tuning; prefer thresholds fixed a priori or tuned on public data.

Every scorer reports ``saw_label`` in the manifest. Scores are attached to rows as
``score_<name>`` float columns. For sentence pairs, field-level scores are aggregated over the
*augmented* fields only (min for similarity/entailment, mean for edit distance).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from views.data import Example
from views.utils import resolve_model_revision, token_edit_distance

REGISTRY: Dict[str, Type["Scorer"]] = {}
FieldPair = Tuple[str, str]  # (original text, view text)


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def build_scorer(cfg: Dict[str, Any], device: str) -> "Scorer":
    name = cfg["name"]
    if name not in REGISTRY:
        raise ValueError(f"unknown scorer {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, device=device)


def augmented_pairs(ex: Example, view: Dict[str, Any]) -> List[FieldPair]:
    out = []
    if "a" in view["augmented_fields"]:
        out.append((ex.text_a, view["text_a"]))
    if "b" in view["augmented_fields"] and ex.text_b is not None:
        out.append((ex.text_b, view["text_b"]))
    return out


class Scorer:
    name = "base"
    saw_label = False
    model_id: Optional[str] = None

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        self.cfg = dict(cfg)
        self.device = device
        self.batch_size = int(cfg.get("batch_size", 32))

    def provenance(self) -> Dict[str, Any]:
        p = {"name": self.name, "saw_label": self.saw_label}
        if self.model_id:
            p.update(model=self.model_id, revision=resolve_model_revision(self.model_id, self.cfg.get("revision")))
        return p

    def score(self, examples: Sequence[Example], views: Sequence[Dict[str, Any]]) -> List[Dict[str, Optional[float]]]:
        """``examples[i]`` is the original of ``views[i]``. Returns one score dict per view."""
        raise NotImplementedError

    # helper: evaluate a pairwise fn on all augmented field pairs, then aggregate per view
    def _fieldwise(self, examples, views, fn, agg) -> List[Optional[float]]:
        pairs, owner = [], []
        for i, (ex, v) in enumerate(zip(examples, views)):
            for fp in augmented_pairs(ex, v):
                pairs.append(fp)
                owner.append(i)
        vals = fn(pairs) if pairs else []
        per_view: List[List[float]] = [[] for _ in views]
        for i, val in zip(owner, vals):
            per_view[i].append(float(val))
        return [agg(x) if x else None for x in per_view]


@register
class LexicalScorer(Scorer):
    """Word-level normalised edit distance (0 = identical) and token length ratio (view / original)."""

    name = "lexical"

    def score(self, examples, views):
        edit = self._fieldwise(examples, views, lambda ps: [token_edit_distance(a, b) for a, b in ps],
                               lambda x: sum(x) / len(x))
        out = []
        for ex, v, e in zip(examples, views, edit):
            n_orig = len(ex.text_a.split()) + len((ex.text_b or "").split())
            n_view = len(v["text_a"].split()) + len((v["text_b"] or "").split())
            out.append({"edit": e, "len_ratio": n_view / max(1, n_orig)})
        return out


@register
class SimilarityScorer(Scorer):
    """Cosine similarity of sentence embeddings, original vs view (min over augmented fields)."""

    name = "similarity"

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        self.model_id = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None

    def _cos(self, pairs: List[FieldPair]) -> List[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device, revision=self.cfg.get("revision"))
        uniq = sorted({t for p in pairs for t in p})
        emb = self._model.encode(uniq, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)
        idx = {t: i for i, t in enumerate(uniq)}
        return [float(emb[idx[a]] @ emb[idx[b]]) for a, b in pairs]

    def score(self, examples, views):
        return [{"sim": s} for s in self._fieldwise(examples, views, self._cos, min)]


class _SeqClsMixin:
    """Thin batched wrapper around an AutoModelForSequenceClassification."""

    def _load_cls(self):
        if getattr(self, "_model", None) is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            rev = self.cfg.get("revision")
            self._tok = AutoTokenizer.from_pretrained(self.model_id, revision=rev)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=rev).to(self.device).eval()
            self.id2label = {int(i): str(l).lower() for i, l in self._model.config.id2label.items()}

    def _probs(self, a: List[str], b: Optional[List[str]] = None):
        import torch

        self._load_cls()
        out = []
        for s in range(0, len(a), self.batch_size):
            args = (a[s:s + self.batch_size],) + ((b[s:s + self.batch_size],) if b is not None else ())
            enc = self._tok(*args, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out.append(torch.softmax(self._model(**enc).logits.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0)


@register
class NLIScorer(_SeqClsMixin, Scorer):
    """Bidirectional entailment between original and view using a public NLI cross-encoder.

    ``entail_fwd`` = P(entail | premise=original, hypothesis=view), ``entail_bwd`` the reverse,
    ``entail`` = min of both (mutual entailment ~ paraphrase), all min-aggregated over fields.
    """

    name = "nli"

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        self.model_id = cfg.get("model", "cross-encoder/nli-deberta-v3-xsmall")
        self._model = None

    def _entail_idx(self) -> int:
        self._load_cls()
        hits = [i for i, l in self.id2label.items() if "entail" in l and "not" not in l]
        if len(hits) != 1:
            raise ValueError(f"cannot find entailment label in {self.id2label}")
        return hits[0]

    def _bidir(self, pairs):
        e = self._entail_idx()
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        fwd = self._probs(a, b)[:, e].tolist()
        bwd = self._probs(b, a)[:, e].tolist()
        return list(zip(fwd, bwd))

    def score(self, examples, views):
        pairs, owner = [], []
        for i, (ex, v) in enumerate(zip(examples, views)):
            for fp in augmented_pairs(ex, v):
                pairs.append(fp)
                owner.append(i)
        vals = self._bidir(pairs) if pairs else []
        per: List[List[Tuple[float, float]]] = [[] for _ in views]
        for i, fb in zip(owner, vals):
            per[i].append(fb)
        out = []
        for x in per:
            if not x:
                out.append({"entail_fwd": None, "entail_bwd": None, "entail": None})
            else:
                f, b = min(v[0] for v in x), min(v[1] for v in x)
                out.append({"entail_fwd": f, "entail_bwd": b, "entail": min(min(v) for v in x)})
        return out


@register
class TaskClassifierScorer(_SeqClsMixin, Scorer):
    """Optional check with a PUBLIC task classifier (must not be trained on the private data).

    * ``task_agree``: 1.0 if argmax(view) == argmax(original) else 0.0 -- label-free.
    * ``task_label_prob`` (only if ``use_label: true``): P(example's own label | view). This sees
      the example's own label -> recorded as ``saw_label: true`` in the manifest.

    ``label_map`` maps dataset label ids to classifier label names (case-insensitive) or ids.
    """

    name = "task"

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        self.model_id = cfg["model"]
        self.saw_label = bool(cfg.get("use_label", False))
        self.label_map = {int(k): v for k, v in (cfg.get("label_map") or {}).items()}
        self.public_data_note = cfg.get("public_data_note", "")
        self._model = None

    def provenance(self):
        return {**super().provenance(), "label_map": self.label_map, "public_data_note": self.public_data_note}

    def _cls_index(self, label: int) -> int:
        target = self.label_map.get(label, label)
        if isinstance(target, int):
            return target
        inv = {l: i for i, l in self.id2label.items()}
        return inv[str(target).lower()]

    def score(self, examples, views):
        if not views:
            return []
        pair = examples[0].text_b is not None
        oa = [e.text_a for e in examples]
        va = [v["text_a"] for v in views]
        ob = [e.text_b for e in examples] if pair else None
        vb = [v["text_b"] for v in views] if pair else None
        p_orig, p_view = self._probs(oa, ob), self._probs(va, vb)
        agree = (p_orig.argmax(-1) == p_view.argmax(-1)).float().tolist()
        out = []
        for i, (ex, ag) in enumerate(zip(examples, agree)):
            d = {"task_agree": ag}
            if self.saw_label and ex.label is not None:
                d["task_label_prob"] = float(p_view[i, self._cls_index(ex.label)])
            out.append(d)
        return out
