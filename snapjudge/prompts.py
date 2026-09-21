"""Prompt construction.

The model is asked to emit an allowed label as its very first answer token. We read the
probability of each label from the logits; nothing is generated.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a decision function inside a software system. You receive a STATE and "
    "one QUESTION about it. Judge only from the state and the question's wording. "
    "Reply with exactly one of the allowed labels and nothing else."
)

NOUL_LABELS = {"yes": ["yes", "Yes"], "no": ["no", "No"]}


def render_value(value) -> str:
    """Text stays text; objects and arrays become readable JSON."""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def state_block(state) -> str:
    return f"<state>\n{render_value(state)}\n</state>\n\n"


def question_header(questions: dict) -> str:
    """All questions of a request, listed before the state (layout "header"). The state is then
    encoded with every question in view, and it is still prefilled only once per request."""
    lines = [f"{i}. {render_value(q['instructions'])}" for i, q in enumerate(questions.values(), 1)]
    return (
        "You will be asked the following questions about the state below, one at a time. "
        "Read the state with all of them in mind.\n" + "\n".join(lines) + "\n\n"
    )


def _variants(label: str) -> list[str]:
    """The label plus a capitalized variant; models like to start a line with a capital."""
    forms = [label]
    cap = label[:1].upper() + label[1:]
    if cap != label:
        forms.append(cap)
    return forms


def question_block(question: dict) -> tuple[str, list[str], dict[str, list[str]]]:
    """Return (question text, options in order, option -> surface forms)."""
    qtype = question["type"]
    instructions = render_value(question["instructions"])
    criteria = question.get("criteria")

    if qtype == "choice":
        options = list(criteria.keys())
        lines = []
        for key, desc in criteria.items():
            lines.append(f"- {key}: {render_value(desc)}" if desc else f"- {key}")
        text = (
            f"QUESTION: {instructions}\n"
            "OPTIONS:\n" + "\n".join(lines) + "\n\n"
            "Reply with exactly one option label: " + ", ".join(options) + "."
        )
        surfaces = _dedupe_surfaces({o: _variants(o) for o in options})
        return text, options, surfaces

    if qtype == "score":
        levels = [str(i) for i in range(len(criteria))]
        lines = [f"{i}: {render_value(desc)}" for i, desc in enumerate(criteria)]
        text = (
            f"QUESTION: {instructions}\n"
            "LEVELS (ordered, lowest first):\n" + "\n".join(lines) + "\n\n"
            f"Reply with exactly one level number (0-{len(levels) - 1})."
        )
        return text, levels, {lvl: [lvl] for lvl in levels}

    if qtype == "noul":
        lines = [f"QUESTION (yes/no): {instructions}"]
        if criteria:
            if criteria.get("true"):
                lines.append(f"yes means: {render_value(criteria['true'])}")
            if criteria.get("false"):
                lines.append(f"no means: {render_value(criteria['false'])}")
        text = "\n".join(lines) + "\n\nReply with exactly yes or no."
        return text, ["yes", "no"], dict(NOUL_LABELS)

    raise ValueError(f"Unknown question type: {qtype}")


def _dedupe_surfaces(surfaces: dict[str, list[str]]) -> dict[str, list[str]]:
    """Each surface form belongs to exactly one option (keys win over variants)."""
    owner = {o: o for o in surfaces}
    out = {}
    for option, forms in surfaces.items():
        keep = []
        for form in forms:
            if owner.setdefault(form, option) == option:
                keep.append(form)
        out[option] = keep
    return out
