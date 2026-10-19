"""Composition route: run one frozen legacy evaluator as the task driver of a
UnifiedRuntime whose bound hosts serve the mechanism APIs through the
canonical store, then write the composition attestation.

Runs inside the evaluator subprocess, after the historical-core import hook
is installed (the bootstrap written by scripts/dev/composition_route_dev.py).
ENGINEERING_ONLY: the runtime uses the TEST_FIXTURES binding policy, so it
reports scientific=False everywhere.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import platform
import sys
import uuid
from pathlib import Path

import torch

import numpy as np

from continuous_brain_protocols import InterventionKind, InterventionSpec

from ..channels import ChannelDecl
from ..runtime import BindingPolicy, RuntimeConfig, UnifiedRuntime
from ..slots import DRIVER_OWNER
from . import boundary as boundary_mod
from . import codec as codec_mod
from . import spec as spec_mod
from .boundary import Boundary, Census, RouteBlocked, install, request_channel
from .spec import SPECS, SPECS_BY_ID, spec_digest

ATTESTATION_FORMAT = "cb-composition-attestation/1"
PACKAGE_FILES = (codec_mod.__file__, boundary_mod.__file__, spec_mod.__file__, __file__)


def _sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scheduler_graph_hash(rt: UnifiedRuntime) -> str:
    rows = [[c.name, c.slot, list(c.reads), list(c.reads_delayed),
             [[d.name, None if d.shape is None else list(d.shape), d.dtype] for d in c.writes]] for c in rt._order]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def build_runtime(bound: list[str], legacy_root: Path, census: Census,
                  extra_modules: tuple = ()) -> tuple[UnifiedRuntime, Boundary]:
    unknown = [m for m in bound if m not in SPECS_BY_ID]
    if unknown:
        raise RouteBlocked(f"unknown mechanisms {unknown}")
    b = Boundary()
    hosts = install(b, SPECS, set(bound), legacy_root, census, extra_modules)
    driver = tuple(ChannelDecl(request_channel(s), DRIVER_OWNER, None, "uint8", intervenable=True,
                               description="encoded legacy API call written by the frozen evaluator (task driver)")
                   for s in sorted(hosts))
    rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES, driver=driver))
    rt.bind_many(list(hosts.values()))
    b.attach(rt, hosts)
    return rt, b


def _mediation(b: Boundary) -> dict:
    host_fw = sum(v for k, v in b.forwards.items() if "|host:" in k)
    driver_fw = sum(v for k, v in b.forwards.items() if k.endswith("|driver"))
    nested = sum(b.nested.values())
    return {"store_mediated_calls": b.calls, "nested_calls_inside_hosts": nested,
            "model_submodule_forwards_in_hosts": host_fw, "model_submodule_forwards_in_driver": driver_fw,
            "class": ("STORE_MEDIATED" if driver_fw == 0 else
                      "PARTLY_DRIVER_DIRECT" if host_fw > 0 else "DRIVER_DIRECT")}


def run_focal(*, evaluator: str, seed_symbol: str, start: int, stop: int, container: str, state_dir: str,
              bound: list[str], focal: str, attestation_path: str, legacy_root: str,
              evaluator_sha256: str, historical_core_digest: str,
              store_interventions: list[dict] | None = None) -> int:
    """store_interventions: engineering positive controls, e.g.
    {"kind": "replace", "target": "autobiography.response", "start_tick": 12, "payload_b64": "..."}"""
    root = Path(legacy_root).resolve()
    census = Census()
    runtime_id = str(uuid.uuid4())
    code = None
    blocked = None
    fatal = None
    rt = b = None
    try:
        spec = importlib.util.spec_from_file_location("cb_g0_frozen_evaluator", evaluator)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        setattr(m, seed_symbol, list(range(start, stop + 1)) if container == "list" else range(start, stop + 1))
        rt, b = build_runtime(bound, root, census, (m,))
        for i, iv in enumerate(store_interventions or []):
            ref = f"control{i}"
            rt.interventions.add_payload(ref, np.frombuffer(base64.b64decode(iv["payload_b64"]), dtype=np.uint8))
            rt.interventions.register(InterventionSpec(InterventionKind(iv["kind"]), iv["target"],
                                                       int(iv["start_tick"]), iv.get("end_tick"), payload_ref=ref,
                                                       note="engineering store-authority positive control"))
        sys.argv = [evaluator, "--state-dir", state_dir]
        try:
            m.main()
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except RouteBlocked as e:
        blocked = str(e)
        code = 3
    except Exception as e:
        code = 1
        fatal = f"{type(e).__name__}: {e}"
    att = {
        "format": ATTESTATION_FORMAT, "status": "ENGINEERING_ONLY", "scientific": False,
        "runtime_instance_id": runtime_id, "focal_mechanism": focal, "bound_mechanisms": sorted(bound),
        "route_blocked": blocked, "fatal_error": fatal, "evaluator_exit_code": code,
        "evaluator": os.path.relpath(evaluator, root), "evaluator_sha256": evaluator_sha256,
        "historical_core_manifest_digest": historical_core_digest,
        "seed_range": [start, stop], "seed_symbol": seed_symbol,
        "codec_version": codec_mod.CODEC_VERSION, "spec_digest": spec_digest(),
        "composition_sources": {Path(p).name: _sha(p) for p in PACKAGE_FILES},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "threads": torch.get_num_threads(), "platform": platform.platform()},
        "census": census.as_dict(),
    }
    if rt is not None:
        att.update({
            "config_hash": rt.config.config_hash(),
            "scheduler_order": [c.name for c in rt._order],
            "scheduler_graph_hash": scheduler_graph_hash(rt),
            "ticks": rt.tick + 1, "store_digest_final": rt.store.digest(), "store_write_log_total": rt.store.log_total,
            "hosts": {slot: {"mechanism_id": h.spec.mechanism_id, **h.stats.as_dict()} for slot, h in sorted(b.hosts.items())},
            "boundary": {"calls": b.calls, "nested_calls": dict(sorted(b.nested.items())),
                         "unbound_calls": dict(sorted(b.unbound_calls.items())),
                         "content_flows": dict(sorted(b.content_flows.items())),
                         "content_flows_meaning": "informative tensor/object content read by the driver from one "
                                                  "host's response that later appeared in another host's request"},
            "model_forwards": dict(sorted(b.forwards.items())),
            "mediation": _mediation(b),
            "api_trace": {slot: {api: {"ticks": h.stats.api_ticks.get(api, []),
                                       "first_response_tick": h.stats.api_first_response[api][0],
                                       "first_response_b64": (base64.b64encode(h.stats.api_first_response[api][1]).decode()
                                                              if len(h.stats.api_first_response[api][1]) <= 65536 else None)}
                                 for api in sorted(h.stats.api_first_response)}
                          for slot, h in sorted(b.hosts.items())},
            "read_report": rt.store.read_report(),
            "first_responses": {slot: {"tick": t, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                       "b64": base64.b64encode(raw).decode() if len(raw) <= 65536 else None}
                                for slot, (t, raw) in sorted(b.first_responses.items())},
            "store_interventions": [{"spec": s_.as_dict()} for s_ in rt.interventions.specs],
            "intervention_applications": len(rt.interventions.records),
            "binding_report": [r.__dict__ for r in rt.binding_report() if r.status == "bound"],
        })
    Path(attestation_path).write_text(json.dumps(att, indent=2, sort_keys=True, default=str) + "\n")
    return int(code)
