"""Unit tests for schema, dedupe, rule views, LLM output cleaning, pair combination, scorers."""
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from views.data import Example, load_examples
from views.generate import combine_pair_views
from views.generators import Generator, RuleGenerator, build_generator, clean_llm_output
from views.schema import ViewRecord, build_schema, validate_table
from views.scorers import LexicalScorer, NLIScorer, augmented_pairs
from views.store import rows_to_table
from views.utils import dedupe, normalize_text, stable_seed, token_edit_distance

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(view_id=0, method="identity", **kw):
    r = ViewRecord("e1", "train", view_id, method, "identity" if method == "identity" else "g", "a b c", None,
                   "" if method == "identity" else "a", 1, NOW).to_row()
    r.update(kw)
    return r


# --- schema ---------------------------------------------------------------------------------
def test_schema_roundtrip_and_score_columns():
    t = rows_to_table([_row(), _row(1, "rule/aeda", score_edit=0.2)])
    assert t.schema.field("score_edit").type == pa.float64()
    assert t.column("score_edit").to_pylist() == [None, 0.2]
    assert t.schema.metadata[b"views_schema_version"] == b"1.0"


def test_schema_rejects_non_identity_view0():
    with pytest.raises(ValueError, match="identity"):
        rows_to_table([_row(0, "rule/aeda")])


def test_schema_rejects_missing_and_extra_columns():
    t = rows_to_table([_row()])
    with pytest.raises(ValueError, match="missing"):
        validate_table(t.drop(["label"]))
    with pytest.raises(ValueError, match="unexpected"):
        validate_table(t.append_column("foo", pa.array([1])))


def test_filtered_schema_has_pad_columns():
    names = build_schema(["score_sim"], filtered=True).names
    assert {"source_view_id", "padded", "pad_source", "score_sim"} <= set(names)


# --- utils ----------------------------------------------------------------------------------
def test_dedupe_drops_original_case_punct_and_empty():
    out = dedupe(["Hello,  World", "hello, world", "", "  ", "Hi there", "hi there", "hi there."], original="HELLO, world")
    assert out == ["Hi there", "hi there."]  # punctuation-only differences are kept (AEDA)


def test_stable_seed_is_deterministic_and_unsalted():
    assert stable_seed(1, "a") == stable_seed(1, "a") != stable_seed(1, "b")
    assert 0 <= stable_seed(0, "x") < 2**31


def test_token_edit_distance():
    assert token_edit_distance("a b c", "a b c") == 0
    assert token_edit_distance("a b c", "a x c") == pytest.approx(1 / 3)
    assert token_edit_distance("", "") == 0
    assert normalize_text(" A,  b! ") == "a, b!"


# --- generators -----------------------------------------------------------------------------
def test_rule_generator_deterministic_unique_and_tagged():
    g = RuleGenerator({"method": "rule"}, seed=3)
    texts, ids = ["the quick brown fox jumps over the lazy dog today"], ["7"]
    a = g.generate(texts, ids, 6, "a")
    b = RuleGenerator({"method": "rule"}, seed=3).generate(texts, ids, 6, "a")
    assert a == b and len(a[0]) == 6
    assert len({t for t, _ in a[0]}) == 6 and texts[0] not in {t for t, _ in a[0]}
    assert {tag for _, tag in a[0]} <= {"rule/aeda", "rule/word_dropout", "rule/word_swap"}
    c = RuleGenerator({"method": "rule"}, seed=4).generate(texts, ids, 6, "a")
    assert c != a


def test_rule_generator_independent_of_batching():
    g = RuleGenerator({"method": "rule"}, seed=0)
    texts, ids = ["one two three four five", "six seven eight nine ten"], ["a", "b"]
    together = g.generate(texts, ids, 3, "a")
    alone = [g.generate([t], [i], 3, "a")[0] for t, i in zip(texts, ids)]
    assert together == alone


def test_rule_generator_short_text_may_shortfall():
    out = RuleGenerator({"method": "rule", "ops": ["word_swap"]}).generate(["hi"], ["0"], 5, "a")
    assert out == [[]]  # cannot swap a single word -> no new views (recorded as shortfall upstream)


def test_provenance_fields_and_generator_id():
    g = build_generator({"method": "rule", "dropout_p": 0.2}, seed=1, device="cpu")
    p = g.provenance()
    assert p["generator_id"].startswith("rule-") and p["seed"] == 1 and p["dropout_p"] == 0.2
    g2 = build_generator({"method": "rule", "dropout_p": 0.3}, seed=1, device="cuda")
    assert g2.generator_id != g.generator_id
    g3 = build_generator({"method": "rule", "dropout_p": 0.2}, seed=1, device="cuda")
    assert g3.generator_id == g.generator_id  # device does not change the id


def test_generator_api_has_no_label_or_pair_access():
    import inspect

    params = list(inspect.signature(Generator.generate).parameters)
    assert params == ["self", "texts", "ids", "k", "field"]


@pytest.mark.parametrize("raw,clean", [
    ("Here's a paraphrase of the text:\n\n\"A lovely film.\"", "A lovely film."),
    ("<think>hmm</think>Paraphrase: The movie was great.\n\nNote: kept tone.", "The movie was great."),
    ("Sure! Here is the rewritten version: “Fine acting.”", "Fine acting."),
    ("A plain\nmultiline answer", "A plain multiline answer"),
])
def test_clean_llm_output(raw, clean):
    assert clean_llm_output(raw) == clean


# --- pair combination -----------------------------------------------------------------------
EX = Example("p", "premise text", "hyp text", 0)


def test_alternate_rewrites_exactly_one_side_and_falls_back():
    pools = {"a": [("P1", "m"), ("P2", "m"), ("P3", "m")], "b": [("H1", "m")]}
    v = combine_pair_views(EX, pools, 4, "alternate")
    assert [x["augmented_fields"] for x in v] == ["a", "b", "a", "a"]
    assert v[1] == {"text_a": "premise text", "text_b": "H1", "augmented_fields": "b", "method": "m"}
    assert all((x["text_a"] == EX.text_a) != (x["text_b"] == EX.text_b) for x in v)


def test_zip_rewrites_both_sides():
    pools = {"a": [("P1", "m"), ("P2", "m")], "b": [("H1", "m")]}
    v = combine_pair_views(EX, pools, 4, "zip")
    assert v == [{"text_a": "P1", "text_b": "H1", "augmented_fields": "ab", "method": "m"}]


# --- scorers --------------------------------------------------------------------------------
def test_lexical_scorer_pair_and_single():
    exs = [Example("1", "a b c d", None, 1), EX]
    views = [{"text_a": "a b x d", "text_b": None, "augmented_fields": "a"},
             {"text_a": "premise text", "text_b": "hyp", "augmented_fields": "b"}]
    s = LexicalScorer({"name": "lexical"}).score(exs, views)
    assert s[0]["edit"] == pytest.approx(0.25) and s[0]["len_ratio"] == 1.0
    assert s[1]["edit"] == pytest.approx(0.5) and s[1]["len_ratio"] == pytest.approx(3 / 4)
    assert augmented_pairs(EX, views[1]) == [("hyp text", "hyp")]


def test_nli_scorer_bidirectional_min_aggregation(monkeypatch):
    sc = NLIScorer({"name": "nli"})
    table = {("P", "P1"): (0.9, 0.6), ("H", "H1"): (0.4, 0.8)}
    monkeypatch.setattr(sc, "_bidir", lambda pairs: [table[p] for p in pairs])
    ex = Example("1", "P", "H", 0)
    out = sc.score([ex, ex], [{"text_a": "P1", "text_b": "H1", "augmented_fields": "ab"},
                              {"text_a": "P", "text_b": "H", "augmented_fields": ""}])
    assert out[0] == {"entail_fwd": 0.4, "entail_bwd": 0.6, "entail": 0.4}
    assert out[1]["entail"] is None


def test_task_scorer_records_label_use():
    from views.scorers import TaskClassifierScorer

    assert TaskClassifierScorer({"name": "task", "model": "x"}).saw_label is False
    assert TaskClassifierScorer({"name": "task", "model": "x", "use_label": True}).saw_label is True


# --- data -----------------------------------------------------------------------------------
def test_inline_loader_column_mapping_and_limit():
    exs, prov = load_examples({"type": "inline", "rows": [{"s": "x", "y": 1}, {"s": "z", "y": 0}],
                               "columns": {"text_a": "s", "label": "y"}}, limit=1)
    assert exs == [Example("0", "x", None, 1)] and prov["num_examples"] == 1
