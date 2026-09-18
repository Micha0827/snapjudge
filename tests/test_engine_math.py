"""Probability logic of the engine, with hand-made token ids and logits instead of a model.

Engine.__new__ skips __init__ (which loads the model); the tree and probability methods only
need `end_id`.
"""

import math

import numpy as np
import pytest

from snapjudge.engine import Engine, _Question

END = 99
VOCAB = 100


@pytest.fixture
def engine():
    e = Engine.__new__(Engine)
    e.end_id = END
    return e


def make_question(engine, paths, options=None, qtype="choice", criteria=None):
    options = options or list(dict.fromkeys(o for o, _ in paths))
    q = _Question("q", qtype, options, criteria, suffix=[], root=engine._build_trie(paths))
    engine._collect_branches(q.root, [], q.branches)
    return q


def logits(**by_token):
    """Logit vector: -inf-like everywhere except the given token ids (t<id>=value)."""
    arr = np.full(VOCAB, -50.0, dtype=np.float32)
    for key, value in by_token.items():
        arr[int(key[1:])] = value
    return arr


# ---------------------------------------------------------------- trie


def test_build_trie_shares_prefixes(engine):
    root = engine._build_trie([("technical", [5, 6]), ("tech_support", [5, 7]), ("sales", [8])])
    assert set(root.children) == {5, 8}
    assert set(root.children[5].children) == {6, 7}
    assert root.children[5].children[6].options == ["technical"]
    assert root.children[8].options == ["sales"]
    assert root.options == []


def test_build_trie_no_duplicate_option_at_node(engine):
    root = engine._build_trie([("a", [1]), ("a", [1])])
    assert root.children[1].options == ["a"]


def test_outgoing_adds_end_token_only_for_label_prefix(engine):
    root = engine._build_trie([("bill", [10]), ("billing", [10, 20])])
    assert engine._outgoing(root) == [10]
    assert engine._outgoing(root.children[10]) == [20, END]  # "bill" ends here, "billing" continues
    assert engine._outgoing(root.children[10].children[20]) == []  # leaf


def test_collect_branches_single_forward_pass(engine):
    q = make_question(engine, [("run", [1]), ("jump", [2]), ("duck", [3])])
    assert [path for _, path in q.branches] == [[]]  # only the root branches


def test_collect_branches_shared_first_token(engine):
    q = make_question(engine, [("technical", [5, 6]), ("tech_support", [5, 7]), ("sales", [8])])
    assert sorted(path for _, path in q.branches) == [[], [5]]


def test_collect_branches_label_prefix(engine):
    q = make_question(engine, [("bill", [10]), ("billing", [10, 20])])
    # root has one child only (no branch), the node after "bill" branches between 20 and END
    assert [path for _, path in q.branches] == [[10]]


# ---------------------------------------------------------------- option probabilities


def softmax(values):
    z = np.exp(np.array(values, dtype=np.float64) - max(values))
    return z / z.sum()


def test_option_probs_root_only(engine):
    q = make_question(engine, [("run", [1]), ("jump", [2]), ("duck", [3])])
    lg = logits(t1=2.0, t2=1.0, t3=0.0, t50=30.0)  # t50 is not allowed and must be ignored
    probs = engine._option_probs(q, {id(q.root): lg}, 1.0)
    expected = softmax([2.0, 1.0, 0.0])
    assert probs["run"] == pytest.approx(expected[0])
    assert probs["jump"] == pytest.approx(expected[1])
    assert probs["duck"] == pytest.approx(expected[2])
    assert sum(probs.values()) == pytest.approx(1.0)


def test_option_probs_temperature(engine):
    q = make_question(engine, [("yes", [1]), ("no", [2])])
    lg = logits(t1=2.0, t2=0.0)
    hot = engine._option_probs(q, {id(q.root): lg}, 2.0)
    assert hot["yes"] == pytest.approx(softmax([1.0, 0.0])[0])
    cold = engine._option_probs(q, {id(q.root): lg}, 0.5)
    assert cold["yes"] > engine._option_probs(q, {id(q.root): lg}, 1.0)["yes"] > hot["yes"]


def test_option_probs_surface_variants_add_up(engine):
    # "billing"/"Billing" and "sales"/"Sales": variants of one option are summed
    q = make_question(engine, [("billing", [1]), ("billing", [2]), ("sales", [3]), ("sales", [4])])
    lg = logits(t1=1.0, t2=1.0, t3=1.0, t4=1.0)
    probs = engine._option_probs(q, {id(q.root): lg}, 1.0)
    assert probs == pytest.approx({"billing": 0.5, "sales": 0.5})


def test_option_probs_shared_first_token_multiplies_along_path(engine):
    q = make_question(engine, [("technical", [5, 6]), ("tech_support", [5, 7]), ("sales", [8])])
    node5 = q.root.children[5]
    at = {
        id(q.root): logits(t5=math.log(3.0), t8=0.0),  # 3:1 -> 0.75 / 0.25
        id(node5): logits(t6=0.0, t7=math.log(2.0)),  # 1:2 -> 1/3 / 2/3
    }
    probs = engine._option_probs(q, at, 1.0)
    assert probs["technical"] == pytest.approx(0.75 / 3)
    assert probs["tech_support"] == pytest.approx(0.75 * 2 / 3)
    assert probs["sales"] == pytest.approx(0.25)


def test_option_probs_label_prefix_bill_billing(engine):
    # "bill" = [10], "billing" = [10, 20]; after token 10 the model either continues with 20
    # or ends the answer (END). Capitalized variants take a separate first token (11).
    paths = [("bill", [10]), ("billing", [10, 20]), ("bill", [11]), ("billing", [11, 20])]
    q = make_question(engine, paths)
    n10, n11 = q.root.children[10], q.root.children[11]
    assert sorted(path for _, path in q.branches) == [[], [10], [11]]
    at = {
        id(q.root): logits(t10=math.log(3.0), t11=0.0),  # lowercase 0.75, capitalized 0.25
        id(n10): logits(t20=0.0, t99=math.log(4.0)),  # after "bill": END 0.8, "ing" 0.2
        id(n11): logits(t20=math.log(4.0), t99=0.0),  # after "Bill": END 0.2, "ing" 0.8
    }
    probs = engine._option_probs(q, at, 1.0)
    assert probs["bill"] == pytest.approx(0.75 * 0.8 + 0.25 * 0.2)
    assert probs["billing"] == pytest.approx(0.75 * 0.2 + 0.25 * 0.8)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_option_probs_single_path_needs_no_logits(engine):
    # Only one allowed continuation everywhere: probability 1 without any forward pass
    q = make_question(engine, [("only", [1, 2, 3])], options=["only"])
    assert q.branches == []
    assert engine._option_probs(q, {}, 1.0) == {"only": 1.0}


def test_option_probs_option_without_paths_gets_zero(engine):
    # An option whose surface forms were all deduplicated away never receives mass
    q = make_question(engine, [("a", [1]), ("b", [2])], options=["a", "b", "c"])
    probs = engine._option_probs(q, {id(q.root): logits(t1=0.0, t2=0.0)}, 1.0)
    assert probs == pytest.approx({"a": 0.5, "b": 0.5, "c": 0.0})


def test_option_probs_shared_leaf_split_evenly(engine):
    # Two options on the same token path share its probability
    q = make_question(engine, [("a", [1]), ("b", [1]), ("c", [2])])
    probs = engine._option_probs(q, {id(q.root): logits(t1=0.0, t2=0.0)}, 1.0)
    assert probs == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.5})


# ---------------------------------------------------------------- confidence and answers


def test_confidence_bounds():
    assert Engine._confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert Engine._confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert Engine._confidence([1.0]) == 1.0


def test_confidence_is_one_minus_normalized_entropy():
    p = [0.1945, 0.784, 0.0215]  # README example: technical vs. billing
    h = -sum(x * math.log(x) for x in p)
    assert Engine._confidence(p) == pytest.approx(1 - h / math.log(3))
    assert Engine._confidence(p) == pytest.approx(0.4614, abs=1e-3)


def test_confidence_never_negative():
    assert Engine._confidence([0.5 + 1e-12, 0.5 - 1e-12]) >= 0.0


def test_answer_choice(engine):
    q = make_question(engine, [("billing", [1]), ("technical", [2])])
    ans = engine._answer(q, {"billing": 0.2, "technical": 0.8})
    assert ans["type"] == "choice"
    assert ans["choice"] == "technical"
    assert ans["probabilities"] == {"billing": 0.2, "technical": 0.8}
    assert 0 < ans["confidence"] < 1


def test_answer_noul(engine):
    q = make_question(engine, [("yes", [1]), ("no", [2])], qtype="noul")
    assert engine._answer(q, {"yes": 0.91234, "no": 0.08766}) == {"type": "noul", "noul": 0.9123}


def test_answer_score_is_expected_level(engine):
    q = make_question(
        engine, [("0", [1]), ("1", [2]), ("2", [3])], qtype="score", criteria=["calm", "civil", "angry"]
    )
    ans = engine._answer(q, {"0": 0.1137, "1": 0.8134, "2": 0.0729})
    assert ans["score"] == pytest.approx(0.8134 + 2 * 0.0729, abs=1e-4)
    assert ans["legend"] == {"0": "calm", "1": "civil", "2": "angry"}
    assert ans["probabilities"] == {"0": 0.1137, "1": 0.8134, "2": 0.0729}
