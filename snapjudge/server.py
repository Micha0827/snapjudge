"""HTTP server with a TypeSafe-compatible API: POST /v1/systemone, GET /v1/models, /game/.

Start with `snapjudge-serve --model <path or Hugging Face id>` (see cli.py), or directly:
    SO_MODEL=<model> uvicorn snapjudge.server:app --port 8724
Optional: SO_NAME (model name in responses), SO_API_KEY (require a bearer token),
          SO_CALIBRATION (per-type temperatures, JSON {model_name: {type: T}}),
          SO_CACHE_LIMIT_GB (cap on MLX buffer cache, default 4), SO_ADAPTER (LoRA adapter directory).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

from .engine import Engine

log = logging.getLogger("snapjudge")
Structured = Union[str, dict, list]
ENGINE: Engine | None = None


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: Structured
    criteria: Union[dict, list, None] = None

    @model_validator(mode="after")
    def check_criteria(self):
        c = self.criteria
        if self.type == "choice" and not (isinstance(c, dict) and len(c) >= 2):
            raise ValueError("choice needs criteria as an object with at least two options")
        if self.type == "score" and not (isinstance(c, list) and len(c) >= 2):
            raise ValueError("score needs criteria as a list with at least two levels")
        if self.type == "noul" and c is not None and not isinstance(c, dict):
            raise ValueError("noul criteria must be an object with true/false")
        return self


class SystemOneRequest(BaseModel):
    state: Structured
    model: str = "local"
    questions: dict[str, Question]
    debug: bool = False
    # Extension over the TypeSafe API: prompt order (auto | state_first | question_first)
    layout: Literal["auto", "state_first", "question_first"] = "auto"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    path = os.environ["SO_MODEL"]
    t = time.perf_counter()
    ENGINE = Engine(path, name=os.environ.get("SO_NAME"), adapter_path=os.environ.get("SO_ADAPTER"))
    log.warning("Loaded %s in %.1f s", ENGINE.name, time.perf_counter() - t)
    yield


app = FastAPI(title="snapjudge", lifespan=lifespan)


def check_auth(authorization: str | None = Header(default=None)):
    # A dependency, so a missing key yields 401 before body validation yields 422
    key = os.environ.get("SO_API_KEY")
    if key and authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


@app.post("/v1/systemone", dependencies=[Depends(check_auth)])
def system_one(req: SystemOneRequest):
    if not req.questions:
        raise HTTPException(status_code=422, detail="questions is empty")
    questions = {qid: q.model_dump() for qid, q in req.questions.items()}
    t = time.perf_counter()
    with ENGINE.lock:
        result = ENGINE.system_one(req.state, questions, debug=req.debug, layout=req.layout)
    log.warning(
        "%d questions, %d tokens, %.0f ms",
        len(questions), result["usage"]["input_tokens"], (time.perf_counter() - t) * 1000,
    )
    return result


@app.get("/v1/models", dependencies=[Depends(check_auth)])
def models():
    return {"models": [{"name": ENGINE.name, "description": f"Local: {ENGINE.model_path}", "release_date": None}]}


# The game: a runner that sends every obstacle as a sentence to /v1/systemone
app.mount("/game", StaticFiles(directory=Path(__file__).resolve().parent / "game", html=True), name="game")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/game/")


@app.get("/health")
def health():
    return {"ok": ENGINE is not None, "model": ENGINE.name if ENGINE else None}
