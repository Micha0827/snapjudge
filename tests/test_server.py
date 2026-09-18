"""Request validation of the HTTP API. No model: the lifespan (which loads it) never runs,
and the endpoint tests use a stub engine."""

import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from snapjudge import server
from snapjudge.server import Question, SystemOneRequest

# ---------------------------------------------------------------- Question


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "choice", "instructions": "Team?", "criteria": {"a": "x", "b": None}},
        {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]},
        {"type": "noul", "instructions": "Urgent?"},
        {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "now", "false": "later"}},
        {"type": "choice", "instructions": {"task": "route", "notes": ["a"]}, "criteria": {"a": 1, "b": 2}},
        {"type": "noul", "instructions": ["step 1", "step 2"]},
    ],
)
def test_question_valid(payload):
    assert Question(**payload).type == payload["type"]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"type": "choice", "instructions": "?", "criteria": {"only": None}}, "at least two options"),
        ({"type": "choice", "instructions": "?", "criteria": ["a", "b"]}, "at least two options"),
        ({"type": "choice", "instructions": "?"}, "at least two options"),
        ({"type": "score", "instructions": "?", "criteria": ["one"]}, "at least two levels"),
        ({"type": "score", "instructions": "?", "criteria": {"0": "low", "1": "high"}}, "at least two levels"),
        ({"type": "noul", "instructions": "?", "criteria": ["yes", "no"]}, "true/false"),
    ],
)
def test_question_invalid_criteria(payload, message):
    with pytest.raises(ValidationError, match=message):
        Question(**payload)


def test_question_unknown_type():
    with pytest.raises(ValidationError):
        Question(type="rank", instructions="?", criteria=["a", "b"])


def test_question_missing_instructions():
    with pytest.raises(ValidationError):
        Question(type="noul")


# ---------------------------------------------------------------- SystemOneRequest


def test_request_defaults():
    req = SystemOneRequest(state="text", questions={"u": {"type": "noul", "instructions": "Urgent?"}})
    assert req.model == "local"
    assert req.debug is False
    assert req.layout == "auto"
    assert isinstance(req.questions["u"], Question)


def test_request_structured_state():
    req = SystemOneRequest(state={"amount": -12.5, "memo": "rent"}, questions={})
    assert req.state == {"amount": -12.5, "memo": "rent"}


@pytest.mark.parametrize("layout", ["auto", "state_first", "question_first"])
def test_request_layouts(layout):
    assert SystemOneRequest(state="x", questions={}, layout=layout).layout == layout


def test_request_invalid_layout():
    with pytest.raises(ValidationError):
        SystemOneRequest(state="x", questions={}, layout="middle")


def test_request_missing_state():
    with pytest.raises(ValidationError):
        SystemOneRequest(questions={"u": {"type": "noul", "instructions": "?"}})


def test_request_invalid_nested_question():
    with pytest.raises(ValidationError, match="at least two options"):
        SystemOneRequest(state="x", questions={"t": {"type": "choice", "instructions": "?", "criteria": {"a": 1}}})


# ---------------------------------------------------------------- endpoint with a stub engine


class StubEngine:
    name = "stub"
    model_path = "stub/path"

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []

    def system_one(self, state, questions, debug=False, layout="auto"):
        self.calls.append((state, questions, debug, layout))
        answers = {qid: {"type": "noul", "noul": 1.0} for qid in questions}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.fixture
def client(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr(server, "ENGINE", stub)
    monkeypatch.delenv("SO_API_KEY", raising=False)
    # Not used as a context manager, so the lifespan (model loading) does not run
    c = TestClient(server.app)
    c.stub = stub
    return c


BODY = {"state": "Server down!", "questions": {"u": {"type": "noul", "instructions": "Urgent?"}}}


def test_endpoint_passes_questions_to_engine(client):
    r = client.post("/v1/systemone", json={**BODY, "layout": "question_first"})
    assert r.status_code == 200
    assert r.json()["answers"]["u"] == {"type": "noul", "noul": 1.0}
    state, questions, debug, layout = client.stub.calls[0]
    assert state == "Server down!"
    assert questions == {"u": {"type": "noul", "instructions": "Urgent?", "criteria": None}}
    assert (debug, layout) == (False, "question_first")


def test_endpoint_rejects_empty_questions(client):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 422
    assert client.stub.calls == []


def test_endpoint_rejects_invalid_question(client):
    body = {"state": "x", "questions": {"t": {"type": "score", "instructions": "?", "criteria": ["one"]}}}
    assert client.post("/v1/systemone", json=body).status_code == 422


def test_endpoint_api_key(client, monkeypatch):
    monkeypatch.setenv("SO_API_KEY", "secret")
    assert client.post("/v1/systemone", json=BODY).status_code == 401
    # Auth is checked before the body: an invalid body without a key is still 401
    assert client.post("/v1/systemone", json={"state": "x"}).status_code == 401
    ok = client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_models_and_health(client):
    assert client.get("/v1/models").json()["models"][0]["name"] == "stub"
    assert client.get("/health").json() == {"ok": True, "model": "stub"}
