"""External semantic/perceptual coprocessor interface (Joint Contract §4).

Hard constraints, enforced here rather than by convention:
  * never the clock: the runtime ticks without waiting on it; components call
    it synchronously and a per-tick call budget caps its use
  * never writes channels: it has no handle on the channel store; responses
    go back to the requesting component only, as data
  * everything logged: every request and response (or refusal) is appended to
    a JSONL log with SHA-256 digests
  * the runtime boots with it disabled; `NullCoprocessor` is the default
  * ablation (G5) swaps any provider for `NullCoprocessor` without touching
    component code, so native and borrowed capability can be separated

Provider adapters (a local model, an API) are optional plugins implementing
`Coprocessor.query`; none ships in this package.
"""

from __future__ import annotations

import abc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from continuous_brain_protocols import canonical_json, sha256_json


@dataclass(frozen=True)
class CoprocessorRequest:
    requester: str
    tick: int
    kind: str
    payload: Mapping[str, Any]

    def digest(self) -> str:
        return sha256_json({"requester": self.requester, "tick": self.tick, "kind": self.kind,
                            "payload": dict(self.payload)})


@dataclass(frozen=True)
class CoprocessorResponse:
    available: bool
    provider: str
    payload: Mapping[str, Any] | None = None
    latency_s: float = 0.0
    refusal: str | None = None


class Coprocessor(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def query(self, request: CoprocessorRequest) -> CoprocessorResponse: ...


class NullCoprocessor(Coprocessor):
    name = "null"

    def query(self, request: CoprocessorRequest) -> CoprocessorResponse:
        return CoprocessorResponse(False, self.name, None, 0.0, "no coprocessor attached")


class CoprocessorBudgetExceeded(RuntimeError):
    pass


@dataclass
class CoprocessorLog:
    path: Path | None = None
    entries: list[dict] = field(default_factory=list)

    def append(self, entry: dict) -> None:
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, sort_keys=True, default=repr) + "\n")   # never fails on odd payloads


class CoprocessorHandle:
    """What a component receives: `query` only, so it cannot enable, re-provision, or re-budget the client."""

    __slots__ = ("_query",)

    def __init__(self, query):
        object.__setattr__(self, "_query", query)

    def __setattr__(self, key, value):
        raise AttributeError("the coprocessor handle is read-only")

    def query(self, requester: str, kind: str, payload: Mapping[str, Any]) -> "CoprocessorResponse":
        return self._query(requester, kind, payload)


def _safe_digest(obj: Any) -> str:
    try:
        return sha256_json(obj)
    except (TypeError, ValueError):
        return sha256_json(repr(obj))


class CoprocessorClient:
    """Owned by the runtime. Enforces enablement, budget, and logging; components get a CoprocessorHandle."""

    def __init__(self, provider: Coprocessor | None = None, enabled: bool = False, max_calls_per_tick: int = 1,
                 log_path: str | os.PathLike | None = None):
        self.provider = provider or NullCoprocessor()
        self.enabled = enabled
        self.max_calls_per_tick = max_calls_per_tick
        self.log = CoprocessorLog(Path(log_path) if log_path else None)
        self._tick = -1
        self._calls_this_tick = 0
        self.total_calls = 0

    def begin_tick(self, tick: int) -> None:
        self._tick, self._calls_this_tick = tick, 0

    def handle(self) -> CoprocessorHandle:
        return CoprocessorHandle(self.query)

    def ablate(self) -> None:
        """G5: remove the provider; components keep calling and get refusals."""
        self.provider = NullCoprocessor()

    def query(self, requester: str, kind: str, payload: Mapping[str, Any]) -> CoprocessorResponse:
        req = CoprocessorRequest(requester, self._tick, kind, dict(payload))
        if not self.enabled:
            resp = CoprocessorResponse(False, self.provider.name, None, 0.0, "coprocessor disabled")
        elif self._calls_this_tick >= self.max_calls_per_tick:
            self._log(req, None, "budget exceeded")
            raise CoprocessorBudgetExceeded(f"{requester}: more than {self.max_calls_per_tick} calls in tick {self._tick}")
        else:
            self._calls_this_tick += 1
            self.total_calls += 1
            t0 = time.perf_counter()
            resp = self.provider.query(req)
            resp = CoprocessorResponse(resp.available, resp.provider, resp.payload, time.perf_counter() - t0,
                                       resp.refusal)
        self._log(req, resp, None)
        return resp

    def _log(self, req: CoprocessorRequest, resp: CoprocessorResponse | None, error: str | None) -> None:
        self.log.append({"tick": req.tick, "requester": req.requester, "kind": req.kind,
                         "request": dict(req.payload), "request_sha256": _safe_digest(dict(req.payload)),
                         "response": None if resp is None else asdict(resp),
                         "response_sha256": None if resp is None else _safe_digest(asdict(resp)),
                         "error": error, "enabled": self.enabled, "provider": self.provider.name})

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "total_calls": self.total_calls, "provider": self.provider.name}

    def load_state_dict(self, state: dict) -> None:
        if state["provider"] != self.provider.name:
            raise ValueError(f"checkpoint used coprocessor {state['provider']!r}, runtime has {self.provider.name!r}")
        self.enabled = bool(state["enabled"])
        self.total_calls = int(state["total_calls"])


def load_log(path: str | os.PathLike) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
