"""Small helpers: stable hashing/seeding, text normalisation, dedupe, provenance info."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Optional


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_hash(obj: Any, n: int = 12) -> str:
    """Deterministic short hash of a JSON-serialisable object (sorted keys)."""
    return sha256_text(json.dumps(obj, sort_keys=True, default=str))[:n]


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts (unlike Python's salted ``hash``)."""
    return int(sha256_text("|".join(str(p) for p in parts))[:8], 16) & 0x7FFFFFFF


def normalize_text(s: str) -> str:
    """Normalisation used for duplicate detection only (never for stored text).

    Case- and whitespace-insensitive. Punctuation is kept on purpose: AEDA-style views differ
    from the original *only* in punctuation and must not be discarded as duplicates.
    """
    return re.sub(r"\s+", " ", s.lower()).strip()


def dedupe(candidates: Iterable[str], original: Optional[str] = None) -> List[str]:
    """Drop empty strings, near-exact duplicates and copies of the original (order-preserving)."""
    seen = set()
    if original is not None:
        seen.add(normalize_text(original))
    out = []
    for c in candidates:
        if c is None:
            continue
        c = c.strip()
        key = normalize_text(c)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def token_edit_distance(a: str, b: str) -> float:
    """Word-level Levenshtein distance normalised by the longer length (0 = identical)."""
    x, y = a.split(), b.split()
    if not x and not y:
        return 0.0
    prev = list(range(len(y) + 1))
    for i, xi in enumerate(x, 1):
        cur = [i] + [0] * len(y)
        for j, yj in enumerate(y, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (xi != yj))
        prev = cur
    return prev[-1] / max(len(x), len(y))


def git_info(path: str | Path = ".") -> dict:
    def _run(*args):
        try:
            return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = _run("rev-parse", "HEAD")
    dirty = bool(_run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit or None, "dirty": dirty}


def is_git_ignored(path: str | Path) -> Optional[bool]:
    """True if ``path`` is ignored by git, False if it would be tracked, None if not in a repo."""
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True, timeout=10)
    except Exception:
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None


def env_versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "datasets", "pyarrow", "sentence_transformers", "vllm"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    return out


def resolve_model_revision(model_id: Optional[str], revision: Optional[str] = None) -> Optional[str]:
    """Resolve a hub model id + revision (branch/tag) to a commit sha; fall back to the given revision."""
    if not model_id:
        return revision
    if Path(model_id).exists():
        return revision or "local"
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:
        return revision


def resolve_dataset_revision(dataset_id: Optional[str], revision: Optional[str] = None) -> Optional[str]:
    if not dataset_id:
        return revision
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_id, revision=revision).sha
    except Exception:
        return revision
