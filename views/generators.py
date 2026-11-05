"""View generators.

Every generator maps *one text field of one example at a time* to up to ``k`` candidate rewrites.
By construction a generator never sees the label, nor the other side of a sentence pair, so
it cannot condition its rewrite on the answer (important for NLI: a hypothesis rewrite must not
"know" the premise or the gold label).

Each generator exposes ``provenance()`` (model id + resolved revision, decoding params, seed,
prompt-template hash, ...) and a ``generator_id`` derived from it, which is stored on every row.
This is what later lets experiments attribute effects to a specific generator distribution
(the "distillation confound": training on LLM views partly distils the generator).

Candidates are returned as ``(text, method_tag)`` tuples, e.g. ``("...", "backtranslation/deu_Latn")``.
"""
from __future__ import annotations

import math
import random
import re
import warnings
from typing import Any, Dict, List, Tuple, Type

from views.utils import dedupe, resolve_model_revision, sha256_text, stable_hash, stable_seed

Candidate = Tuple[str, str]
REGISTRY: Dict[str, Type["Generator"]] = {}


def register(cls):
    REGISTRY[cls.method] = cls
    return cls


def build_generator(cfg: Dict[str, Any], seed: int, device: str) -> "Generator":
    method = cfg["method"]
    if method not in REGISTRY:
        raise ValueError(f"unknown generator method {method!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[method](cfg, seed=seed, device=device)


class Generator:
    method = "base"
    #: whether the generator is a (learned) model whose output distribution could be distilled
    model_based = False

    def __init__(self, cfg: Dict[str, Any], seed: int = 0, device: str = "cpu"):
        self.cfg = dict(cfg)
        self.seed = int(cfg.get("seed", seed))
        self.device = device

    # -- provenance ---------------------------------------------------------------------------
    def _provenance_fields(self) -> Dict[str, Any]:
        return {}

    def provenance(self) -> Dict[str, Any]:
        core = {"method": self.method, "seed": self.seed, "model_based": self.model_based, **self._provenance_fields()}
        # generator_id depends only on what determines the output distribution (not on device)
        return {"generator_id": f"{self.method}-{stable_hash(core, 10)}", **core}

    @property
    def generator_id(self) -> str:
        if not hasattr(self, "_gid"):
            self._gid = self.provenance()["generator_id"]
        return self._gid

    # -- generation ---------------------------------------------------------------------------
    def generate(self, texts: List[str], ids: List[str], k: int, field: str) -> List[List[Candidate]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------
@register
class RuleGenerator(Generator):
    """Cheap rule-based views: AEDA punctuation insertion, word dropout, adjacent word swap.

    No external dependencies. Per-view RNG is derived from (seed, generator_id-relevant cfg,
    example id, field, view index), so output is reproducible and independent of batching.
    """

    method = "rule"
    PUNCT = [".", ";", "?", ":", "!", ","]
    OPS = ("aeda", "word_dropout", "word_swap")

    def __init__(self, cfg, seed=0, device="cpu"):
        super().__init__(cfg, seed, device)
        self.ops = list(cfg.get("ops", ["aeda", "word_dropout", "word_swap"]))
        bad = set(self.ops) - set(self.OPS)
        if bad:
            raise ValueError(f"unknown rule ops: {bad}")
        self.dropout_p = float(cfg.get("dropout_p", 0.1))
        self.swap_n = int(cfg.get("swap_n", 1))
        self.max_attempts_factor = int(cfg.get("max_attempts_factor", 4))

    def _provenance_fields(self):
        return {"ops": self.ops, "dropout_p": self.dropout_p, "swap_n": self.swap_n}

    @classmethod
    def aeda(cls, words: List[str], rng: random.Random) -> List[str]:
        """AEDA (Karimi et al., 2021): insert 1..max(1, n/3) random punctuation marks."""
        out = list(words)
        n_ins = rng.randint(1, max(1, len(words) // 3))
        for _ in range(n_ins):
            out.insert(rng.randint(0, len(out)), rng.choice(cls.PUNCT))
        return out

    def word_dropout(self, words, rng):
        if len(words) <= 1:
            return list(words)
        kept = [w for w in words if rng.random() >= self.dropout_p]
        if len(kept) == len(words):  # force at least one deletion
            del kept[rng.randrange(len(kept))]
        return kept or [rng.choice(words)]

    def word_swap(self, words, rng):
        out = list(words)
        if len(out) < 2:
            return out
        for _ in range(self.swap_n):
            i = rng.randrange(len(out) - 1)
            out[i], out[i + 1] = out[i + 1], out[i]
        return out

    def generate(self, texts, ids, k, field):
        results = []
        for text, ex_id in zip(texts, ids):
            words = text.split()
            raw: List[Candidate] = []
            for attempt in range(k * self.max_attempts_factor):
                op = self.ops[attempt % len(self.ops)]
                rng = random.Random(stable_seed(self.seed, self.generator_id, ex_id, field, attempt))
                new = getattr(self, op)(words, rng)
                raw.append((" ".join(new), f"rule/{op}"))
                kept = dedupe([t for t, _ in raw], original=text)
                if len(kept) >= k:
                    break
            results.append(_dedupe_candidates(raw, text)[:k])
        return results


@register
class MockGenerator(Generator):
    """Deterministic stand-in for a model-based generator (tests / offline smoke runs only)."""

    method = "mock"
    model_based = True

    def _provenance_fields(self):
        return {"note": "mock outputs, not a real model"}

    def generate(self, texts, ids, k, field):
        out = []
        for text, ex_id in zip(texts, ids):
            rng = random.Random(stable_seed(self.seed, ex_id, field))
            cands = []
            for i in range(k + 1):  # one duplicate to exercise dedupe
                words = text.split()
                if words:
                    j = rng.randrange(len(words))
                    words[j] = words[j].upper() if i % 2 == 0 else words[j] + "s"
                cands.append((" ".join(words) + f" [m{i // 2}]", "mock"))
            out.append(_dedupe_candidates(cands, text)[:k])
        return out


def _dedupe_candidates(cands: List[Candidate], original: str) -> List[Candidate]:
    kept = set(dedupe([t for t, _ in cands], original=original))
    out, used = [], set()
    for t, tag in cands:
        t = t.strip()
        if t in kept and t not in used:
            used.add(t)
            out.append((t, tag))
    return out


# ---------------------------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful editor who rewrites text without changing what it says."
)
DEFAULT_USER_TEMPLATE = (
    "Paraphrase the text below. Keep its meaning, sentiment, stance, facts, negations, numbers, "
    "units and domain terminology exactly as they are, but vary the wording and sentence structure. "
    "Do not add or remove information. Output only the paraphrase, nothing else.\n\n"
    "Text: {text}"
)
_PREFIX_RE = re.compile(
    r"^\s*(sure[,!.]?\s*)?(here(?:'s| is) (?:a|the|one|my) (?:possible )?(?:paraphrase|rewrite|rewritten version)[^:\n]*:"
    r"|paraphrased?(?: text| version)?:|rewritten(?: text| version)?:|output:)\s*",
    re.IGNORECASE,
)


def clean_llm_output(s: str) -> str:
    """Strip reasoning blocks, chatty prefixes and wrapping quotes; keep the first paragraph."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.replace("<think>", "").strip()
    s = _PREFIX_RE.sub("", s, count=1).strip()
    s = re.split(r"\n\s*\n", s, maxsplit=1)[0]
    s = re.sub(r"\s*\n\s*", " ", s).strip()
    for q1, q2 in (('"', '"'), ("“", "”"), ("'", "'"), ("«", "»")):
        if len(s) >= 2 and s.startswith(q1) and s.endswith(q2):
            s = s[1:-1].strip()
    return s


@register
class LLMParaphraseGenerator(Generator):
    """Sampling-based paraphrases from a local instruction-tuned LLM (transformers or vLLM)."""

    method = "llm_paraphrase"
    model_based = True

    def __init__(self, cfg, seed=0, device="cpu"):
        super().__init__(cfg, seed, device)
        self.model_id = cfg["model"]
        self.revision = cfg.get("revision")
        self.system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.user_template = cfg.get("user_template", DEFAULT_USER_TEMPLATE)
        if "{text}" not in self.user_template:
            raise ValueError("user_template must contain '{text}'")
        self.decoding = {
            "temperature": float(cfg.get("temperature", 0.9)),
            "top_p": float(cfg.get("top_p", 0.95)),
            "max_new_tokens": int(cfg.get("max_new_tokens", 128)),
            "oversample": float(cfg.get("oversample", 1.5)),
        }
        self.batch_size = int(cfg.get("batch_size", 8))
        self.chat_template_kwargs = dict(cfg.get("chat_template_kwargs", {}))
        self.backend_req = cfg.get("backend", "auto")
        self._resolved_revision = resolve_model_revision(self.model_id, self.revision)
        self._model = self._tok = self._llm = None
        self.backend = None
        self._planned_backend = self._pick_backend()

    def _provenance_fields(self):
        return {
            "model": self.model_id,
            "revision": self._resolved_revision,
            "decoding": self.decoding,
            "prompt_template_sha256": sha256_text(self.system_prompt + "\x00" + self.user_template),
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "chat_template_kwargs": self.chat_template_kwargs,
            # transformers sampling depends on batch composition; vLLM uses per-request seeds
            "batch_size": self.batch_size,
            "backend": self._planned_backend,
        }

    def _pick_backend(self) -> str:
        if self.backend_req == "transformers":
            return "transformers"
        try:
            import torch
            import vllm  # noqa: F401

            if torch.cuda.is_available():
                return "vllm"
        except Exception:
            pass
        if self.backend_req == "vllm":
            warnings.warn("vLLM requested but unavailable (needs vllm + CUDA); falling back to transformers")
        return "transformers"

    def _load(self):
        if self.backend is not None:
            return
        from transformers import AutoTokenizer

        self.backend = self._planned_backend
        self._tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        if self._tok.chat_template is None:
            raise ValueError(f"{self.model_id} has no chat template; use an instruction-tuned model")
        if self.backend == "vllm":
            from vllm import LLM

            self._llm = LLM(model=self.model_id, revision=self.revision, seed=self.seed)
        else:
            import torch
            from transformers import AutoModelForCausalLM

            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision, dtype=dtype)
            except TypeError:  # older transformers
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision, torch_dtype=dtype)
            self._model.to(self.device).eval()
            self._tok.padding_side = "left"
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token

    def build_prompt(self, text: str) -> str:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": self.user_template.format(text=text)})
        return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)

    def _sample(self, prompts: List[str], ids: List[str], n: int, field: str) -> List[List[str]]:
        d = self.decoding
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = [
                SamplingParams(n=n, temperature=d["temperature"], top_p=d["top_p"], max_tokens=d["max_new_tokens"],
                               seed=stable_seed(self.seed, ex_id, field))
                for ex_id in ids
            ]
            outs = self._llm.generate(prompts, params, use_tqdm=False)
            return [[o.text for o in out.outputs] for out in outs]

        import torch

        torch.manual_seed(stable_seed(self.seed, field, ids[0]))
        enc = self._tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            gen = self._model.generate(
                **enc, do_sample=True, temperature=d["temperature"], top_p=d["top_p"],
                max_new_tokens=d["max_new_tokens"], num_return_sequences=n, pad_token_id=self._tok.pad_token_id,
            )
        texts = self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [texts[i * n:(i + 1) * n] for i in range(len(prompts))]

    def generate(self, texts, ids, k, field):
        self._load()
        n = max(k, math.ceil(k * self.decoding["oversample"]))
        results = []
        for s in range(0, len(texts), self.batch_size):
            bt, bi = texts[s:s + self.batch_size], ids[s:s + self.batch_size]
            samples = self._sample([self.build_prompt(t) for t in bt], bi, n, field)
            for text, outs in zip(bt, samples):
                cands = [(clean_llm_output(o), "llm_paraphrase") for o in outs]
                results.append(_dedupe_candidates(cands, text)[:k])
        return results


# ---------------------------------------------------------------------------------------------
@register
class BackTranslationGenerator(Generator):
    """Round-trip translation through pivot languages (NLLB-200 or Opus-MT pairs).

    Forward (en -> pivot) uses beam search; backward (pivot -> en) draws ``samples_per_pivot``
    candidates (beam if 1, sampling otherwise). One method tag per pivot, e.g. ``backtranslation/deu_Latn``.
    """

    method = "backtranslation"
    model_based = True

    def __init__(self, cfg, seed=0, device="cpu"):
        super().__init__(cfg, seed, device)
        self.backend = cfg.get("backend", "nllb")
        if self.backend not in ("nllb", "opus"):
            raise ValueError("backtranslation.backend must be 'nllb' or 'opus'")
        self.model_id = cfg.get("model", "facebook/nllb-200-distilled-600M") if self.backend == "nllb" else None
        self.src = cfg.get("src_lang", "eng_Latn" if self.backend == "nllb" else "en")
        self.pivots = list(cfg.get("pivots", ["deu_Latn", "fra_Latn", "zho_Hans", "rus_Cyrl"]
                                   if self.backend == "nllb" else ["de", "fr", "zh", "ru"]))
        self.opus_template = cfg.get("opus_template", "Helsinki-NLP/opus-mt-{src}-{tgt}")
        self.decoding = {
            "num_beams": int(cfg.get("num_beams", 4)),
            "samples_per_pivot": int(cfg.get("samples_per_pivot", 1)),
            "temperature": float(cfg.get("temperature", 0.8)),
            "top_p": float(cfg.get("top_p", 0.95)),
            "max_length": int(cfg.get("max_length", 256)),
        }
        self.batch_size = int(cfg.get("batch_size", 16))
        self._cache: Dict[str, Any] = {}
        ids = [self.model_id] if self.model_id else [
            self.opus_template.format(src=a, tgt=b) for p in self.pivots for a, b in ((self.src, p), (p, self.src))
        ]
        self._revisions = {m: resolve_model_revision(m, cfg.get("revision")) for m in ids}

    def _provenance_fields(self):
        return {"backend": self.backend, "models": self._revisions, "src_lang": self.src,
                "pivots": self.pivots, "decoding": self.decoding, "batch_size": self.batch_size}

    def _load(self, model_id):
        if model_id not in self._cache:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_id, revision=self.cfg.get("revision"))
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=self.cfg.get("revision")).to(self.device).eval()
            self._cache[model_id] = (tok, model)
        return self._cache[model_id]

    def _translate(self, texts, src, tgt, n, sample, seed):
        import torch

        if self.backend == "nllb":
            tok, model = self._load(self.model_id)
            tok.src_lang = src
            extra = {"forced_bos_token_id": tok.convert_tokens_to_ids(tgt)}
        else:
            tok, model = self._load(self.opus_template.format(src=src, tgt=tgt))
            extra = {}
        d = self.decoding
        out = []
        for s in range(0, len(texts), self.batch_size):
            torch.manual_seed(stable_seed(seed, s))
            enc = tok(texts[s:s + self.batch_size], return_tensors="pt", padding=True, truncation=True,
                      max_length=d["max_length"]).to(self.device)
            kw = dict(do_sample=True, temperature=d["temperature"], top_p=d["top_p"]) if sample else dict(num_beams=max(d["num_beams"], n))
            with torch.no_grad():
                gen = model.generate(**enc, num_return_sequences=n, max_length=d["max_length"], **kw, **extra)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            out += [dec[i * n:(i + 1) * n] for i in range(len(dec) // n)]
        return out

    def generate(self, texts, ids, k, field):
        n_back = self.decoding["samples_per_pivot"]
        per_text: List[List[Candidate]] = [[] for _ in texts]
        for p in self.pivots:
            fwd = [o[0] for o in self._translate(texts, self.src, p, 1, False, stable_seed(self.seed, field, p, "fwd"))]
            back = self._translate(fwd, p, self.src, n_back, n_back > 1, stable_seed(self.seed, field, p, "bwd"))
            for i, outs in enumerate(back):
                per_text[i] += [(o, f"backtranslation/{p}") for o in outs]
        return [_dedupe_candidates(c, t)[:k] for c, t in zip(per_text, texts)]
