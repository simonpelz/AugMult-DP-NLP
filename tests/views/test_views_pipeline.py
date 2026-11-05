"""End-to-end tests (no network): generate -> store -> filter, provenance and manifests."""
import json

import pytest

from views import filter as vfilter
from views import generate as vgen
from views.filter import filter_rows
from views.store import load_grouped, read_store, views_per_example


@pytest.fixture
def no_network(monkeypatch):
    import views.utils as u

    monkeypatch.setattr(u, "resolve_model_revision", lambda m, r=None: r)
    monkeypatch.setattr(u, "resolve_dataset_revision", lambda d, r=None: r)


def test_generate_single_text(tmp_path, sst_cfg, no_network):
    out = vgen.main(["--config", str(sst_cfg), "--output-dir", str(tmp_path / "out")])
    table, man = read_store(out)
    rows = table.to_pylist()
    assert man["kind"] == "views" and man["dataset"]["num_examples"] == 3
    # identity + 4 rule + 2 mock per example
    assert set(views_per_example(table).values()) == {7}
    for ex in {r["example_id"] for r in rows}:
        g = sorted((r for r in rows if r["example_id"] == ex), key=lambda r: r["view_id"])
        assert [r["view_id"] for r in g] == list(range(7))
        assert g[0]["method"] == "identity" and g[0]["augmented_fields"] == ""
        assert all(r["label"] == g[0]["label"] for r in g)
        assert len({r["text_a"] for r in g}) == 7  # deduped, none equal to original
    # provenance
    gids = {p["generator_id"] for p in man["generators"]}
    assert gids == {r["generator_id"] for r in rows} - {"identity"}
    assert all("seed" in p and "method" in p for p in man["generators"])
    assert man["scorers"] == [{"name": "lexical", "saw_label": False}]
    assert man["scorers_that_saw_labels"] == []
    assert man["shortfall_views"] and man["summary"]["counts"]["rows"] == 21
    assert man["summary"]["label_flip_rate_estimate"]["mock"]["value"] is None  # no NLI/task scorer
    assert "score_edit" in table.column_names and "score_len_ratio" in table.column_names
    assert man["config_hash"] in out.name and json.dumps(man)  # serialisable


def test_generate_is_reproducible(tmp_path, sst_cfg, no_network):
    a = read_store(vgen.main(["--config", str(sst_cfg), "--output-dir", str(tmp_path / "a")]))[0]
    b = read_store(vgen.main(["--config", str(sst_cfg), "--output-dir", str(tmp_path / "b")]))[0]
    cols = ["example_id", "view_id", "method", "generator_id", "text_a", "score_edit"]
    assert a.select(cols).equals(b.select(cols))


def test_generate_pairs_alternate(tmp_path, nli_cfg, no_network):
    out = vgen.main(["--config", str(nli_cfg), "--output-dir", str(tmp_path / "o")])
    table, _ = read_store(out)
    rows = [r for r in table.to_pylist() if r["view_id"] > 0]
    assert rows and {r["augmented_fields"] for r in rows} == {"a", "b"}
    orig = {r["example_id"]: r for r in table.to_pylist() if r["view_id"] == 0}
    for r in rows:
        o = orig[r["example_id"]]
        changed = {"a": r["text_a"] != o["text_a"], "b": r["text_b"] != o["text_b"]}
        assert [f for f, c in changed.items() if c] == [r["augmented_fields"]]


def test_refuses_overwrite(tmp_path, sst_cfg, no_network):
    vgen.main(["--config", str(sst_cfg), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        vgen.main(["--config", str(sst_cfg), "--output-dir", str(tmp_path)])


def _rows(scores_per_view, ex="e"):
    rows = [{"example_id": ex, "view_id": 0, "method": "identity", "generator_id": "identity", "text_a": "orig",
             "text_b": None, "augmented_fields": "", "label": 1, "score_entail": None}]
    for i, s in enumerate(scores_per_view, 1):
        rows.append({**rows[0], "view_id": i, "method": "llm_paraphrase", "generator_id": "g",
                     "text_a": f"v{i}", "augmented_fields": "a", "score_entail": s})
    return rows


def test_filter_threshold_and_exact_k():
    out, st = filter_rows(_rows([0.9, 0.2, 0.95, None]), k=3, thresholds={"min_entail": 0.8})
    assert [r["text_a"] for r in out] == ["orig", "v1", "v3"]
    assert [r["source_view_id"] for r in out] == [0, 1, 3] and [r["view_id"] for r in out] == [0, 1, 2]
    assert not any(r["padded"] for r in out) and st["examples_padded"] == 0
    assert st["pass_rate_by_method"]["llm_paraphrase"] == {"seen": 4, "passed": 2, "rate": 0.5}


def test_filter_pad_resample_then_identity_fallback():
    out, st = filter_rows(_rows([0.9, 0.1]) + _rows([0.1], ex="f"), k=4, thresholds={"min_entail": 0.8})
    e = [r for r in out if r["example_id"] == "e"]
    f = [r for r in out if r["example_id"] == "f"]
    assert len(e) == len(f) == 4
    assert [r["pad_source"] for r in e] == [None, None, "resample", "resample"] and e[2]["text_a"] == "v1"
    assert [r["pad_source"] for r in f] == [None, "identity", "identity", "identity"]
    assert all(r["text_a"] == "orig" for r in f)
    assert st["pad_rows"] == {"resample": 2, "identity": 3} and st["examples_padded"] == 2


def test_filter_pad_identity_and_random_select_seeded():
    out, _ = filter_rows(_rows([0.9]), k=3, thresholds={}, pad="identity")
    assert [r["pad_source"] for r in out] == [None, None, "identity"]
    a, _ = filter_rows(_rows([0.9] * 6), k=3, thresholds={}, select="random", seed=1)
    b, _ = filter_rows(_rows([0.9] * 6), k=3, thresholds={}, select="random", seed=1)
    assert a == b


def test_filter_cli_end_to_end(tmp_path, sst_cfg, no_network):
    src = vgen.main(["--config", str(sst_cfg), "--output-dir", str(tmp_path / "g")])
    dst = vfilter.main(["--input", str(src), "--k", "5", "--min-edit", "0.01", "--methods", "rule"])
    table, man = read_store(dst)
    assert man["kind"] == "filtered_views" and man["parent"]["config_hash"]
    assert set(views_per_example(table).values()) == {5}
    assert {r["method"].split("/")[0] for r in table.to_pylist()} <= {"identity", "rule"}
    assert len(load_grouped(dst, k=5)) == 3
    with pytest.raises(SystemExit):  # unknown score column
        vfilter.main(["--input", str(src), "--k", "3", "--min-entail", "0.5", "--output", str(tmp_path / "x")])
    with pytest.raises(SystemExit):  # refuse to re-filter a filtered store
        vfilter.main(["--input", str(dst), "--k", "3", "--output", str(tmp_path / "y")])
