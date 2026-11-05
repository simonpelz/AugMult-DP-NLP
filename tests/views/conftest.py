import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SST_ROWS = [
    {"sentence": "a gorgeous , witty , seductive movie .", "label": 1, "idx": 10},
    {"sentence": "the plot is nothing but boilerplate clichés from start to finish", "label": 0, "idx": 11},
    {"sentence": "it 's a charming and often affecting journey .", "label": 1, "idx": 12},
]
NLI_ROWS = [
    {"premise": "The man is playing a guitar on stage tonight.", "hypothesis": "A man performs music.", "label": 0, "pid": "p1"},
    {"premise": "Two dogs run across a snowy field.", "hypothesis": "The animals are asleep indoors.", "label": 2, "pid": "p2"},
]


def write_cfg(tmp_path, cfg_text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg_text)
    return p


@pytest.fixture
def sst_cfg(tmp_path):
    import yaml

    cfg = {
        "name": "t_sst", "seed": 0, "k": 4,
        "dataset": {"type": "inline", "rows": SST_ROWS, "columns": {"text_a": "sentence", "label": "label", "id": "idx"}},
        "generators": [{"method": "rule"}, {"method": "mock", "k": 2}],
        "scorers": [{"name": "lexical"}],
    }
    return write_cfg(tmp_path, yaml.safe_dump(cfg))


@pytest.fixture
def nli_cfg(tmp_path):
    import yaml

    cfg = {
        "name": "t_nli", "seed": 0, "k": 4, "pair_combine": "alternate",
        "dataset": {"type": "inline", "rows": NLI_ROWS,
                    "columns": {"text_a": "premise", "text_b": "hypothesis", "label": "label", "id": "pid"}},
        "generators": [{"method": "mock"}],
        "scorers": [{"name": "lexical"}],
    }
    return write_cfg(tmp_path, yaml.safe_dump(cfg))
