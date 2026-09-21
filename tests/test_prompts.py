"""Prompt construction: no model, no tokenizer."""

import pytest

from snapjudge.prompts import NOUL_LABELS, _dedupe_surfaces, question_block, question_header, render_value, state_block


def test_render_value_strips_text():
    assert render_value("  hello \n") == "hello"


def test_render_value_structured_as_readable_json():
    out = render_value({"name": "Müller", "items": [1, 2]})
    assert '"name": "Müller"' in out  # ensure_ascii=False keeps umlauts
    assert "\n  " in out  # indented
    assert render_value([1, "a"]) == '[\n  1,\n  "a"\n]'


def test_state_block():
    assert state_block(" x ") == "<state>\nx\n</state>\n\n"


def test_choice_block():
    q = {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None},
    }
    text, options, surfaces = question_block(q)
    assert options == ["billing", "technical", "sales"]
    assert "QUESTION: Which team?" in text
    assert "- billing: Payments" in text
    assert "- sales\n" in text  # no description, no colon
    assert text.endswith("Reply with exactly one option label: billing, technical, sales.")
    assert surfaces == {
        "billing": ["billing", "Billing"],
        "technical": ["technical", "Technical"],
        "sales": ["sales", "Sales"],
    }


def test_choice_block_capitalized_key_has_no_duplicate_variant():
    _, _, surfaces = question_block({"type": "choice", "instructions": "?", "criteria": {"A": None, "b": None}})
    assert surfaces == {"A": ["A"], "b": ["b", "B"]}


def test_choice_block_structured_instructions():
    text, _, _ = question_block({"type": "choice", "instructions": {"task": "route"}, "criteria": {"a": 1, "b": 2}})
    assert '"task": "route"' in text
    assert "- a: 1" in text


def test_score_block():
    text, options, surfaces = question_block(
        {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]}
    )
    assert options == ["0", "1", "2"]
    assert surfaces == {"0": ["0"], "1": ["1"], "2": ["2"]}
    assert "LEVELS (ordered, lowest first):\n0: calm\n1: annoyed\n2: furious" in text
    assert text.endswith("Reply with exactly one level number (0-2).")


def test_noul_block_without_criteria():
    text, options, surfaces = question_block({"type": "noul", "instructions": "Urgent?"})
    assert options == ["yes", "no"]
    assert surfaces == NOUL_LABELS
    assert surfaces is not NOUL_LABELS  # a copy, callers may not mutate the constant
    assert text == "QUESTION (yes/no): Urgent?\n\nReply with exactly yes or no."


def test_noul_block_with_criteria():
    text, _, _ = question_block(
        {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "deadline today", "false": "no deadline"}}
    )
    assert "yes means: deadline today" in text
    assert "no means: no deadline" in text


def test_noul_block_partial_criteria():
    text, _, _ = question_block({"type": "noul", "instructions": "Urgent?", "criteria": {"true": "now"}})
    assert "yes means: now" in text
    assert "no means" not in text


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown question type"):
        question_block({"type": "rank", "instructions": "?"})


def test_dedupe_keys_win_over_variants():
    # "Yes" is its own option, so the capitalized variant of "yes" must not also count for "yes"
    out = _dedupe_surfaces({"yes": ["yes", "Yes"], "Yes": ["Yes"]})
    assert out == {"yes": ["yes"], "Yes": ["Yes"]}


def test_dedupe_first_option_wins_shared_variant():
    out = _dedupe_surfaces({"a": ["a", "X"], "b": ["b", "X"]})
    assert out == {"a": ["a", "X"], "b": ["b"]}


def test_dedupe_leaves_disjoint_forms_alone():
    surfaces = {"run": ["run", "Run"], "jump": ["jump", "Jump"]}
    assert _dedupe_surfaces(surfaces) == surfaces


def test_question_header_lists_all_questions_in_order():
    qs = {
        "urgent": {"type": "noul", "instructions": "  Is this urgent? "},
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "", "tech": ""}},
    }
    header = question_header(qs)
    assert header.startswith("You will be asked the following questions about the state below")
    assert "1. Is this urgent?\n2. Which team?" in header
    assert "billing" not in header  # only the questions, not their options
    assert header.endswith("\n\n")

