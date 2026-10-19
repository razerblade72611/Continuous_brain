"""Stateful-host contract: the store holds the individual's state (v0.38.1.dev7, ENGINEERING_ONLY)."""

import numpy as np
import pytest

from continuous_brain_protocols import InterventionKind, InterventionSpec
from continuous_brain_snn.snapshot import SnapshotStore
from continuous_brain_unified.channels import ChannelDecl
from continuous_brain_unified.composition.boundary import Boundary, RouteBlocked, request_channel
from continuous_brain_unified.composition.codec import encode
from continuous_brain_unified.integration.hosts import StatefulHost, state_channel
from continuous_brain_unified.integration.life import Client
from continuous_brain_unified.runtime import BindingPolicy, RuntimeConfig, UnifiedRuntime
from continuous_brain_unified.slots import DRIVER_OWNER


class Tally(StatefulHost):
    """A minimal stateful host: an append-only list of numbers in the autobiography slot."""
    slot = "autobiography"
    mechanism_id = "test_tally"
    canonical = (("autobiography.record", (27,)), ("autobiography.discrepancy", (2,)))

    def initial_state(self):
        return []

    def apis(self):
        return {"push": lambda st, x: (st + [x], len(st) + 1), "total": lambda st: (st, sum(st)),
                "boom": self._boom}

    def _boom(self, st):
        raise ValueError("legacy failure")

    def project(self, st):
        return {"autobiography.discrepancy": np.array([float(sum(st)), float(len(st))])} if st else {}


def make(interventions=()):
    b = Boundary()
    host = Tally(b, {})
    driver = (ChannelDecl(request_channel("autobiography"), DRIVER_OWNER, None, "uint8"),)
    rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES, driver=driver))
    rt.bind_many([host])
    for i, (kind, start, end, payload) in enumerate(interventions):
        ref = None
        if payload is not None:
            ref = f"p{i}"
            rt.interventions.add_payload(ref, np.frombuffer(encode(payload).data, dtype=np.uint8))
        rt.interventions.register(InterventionSpec(kind, state_channel("autobiography"), start, end, payload_ref=ref))
    b.attach(rt, {"autobiography": host})
    return rt, host, Client(b)


def test_state_lives_in_the_store_and_persists_across_ticks():
    rt, host, c = make()
    assert c("autobiography", "push", 2) == 1
    assert c("autobiography", "push", 3) == 2
    assert c("autobiography", "total") == 5
    assert not hasattr(host, "_state")                  # nothing kept in Python between ticks
    reads = {(r["reader"], r["channel"]): r for r in rt.store.read_report()}
    assert reads[("autobiography", state_channel("autobiography"))]["nonempty_reads"] == 2   # delayed reads


def test_store_interventions_change_the_individual():
    # ERASE after the second push (tick 1): the third call sees an empty history
    rt, host, c = make([(InterventionKind.ERASE, 1, 2, None)])
    c("autobiography", "push", 2); c("autobiography", "push", 3)
    assert c("autobiography", "total") == 0
    # REPLACE with a donor history at tick 0
    rt, host, c = make([(InterventionKind.REPLACE, 0, 1, [10, 20])])
    c("autobiography", "push", 1)
    assert c("autobiography", "total") == 30
    # FREEZE from tick 1: pushes during the window do not persist
    rt, host, c = make([(InterventionKind.FREEZE, 1, 3, None)])
    c("autobiography", "push", 1); c("autobiography", "push", 2); c("autobiography", "push", 4)
    assert c("autobiography", "total") == 1


def test_legacy_errors_travel_back_and_state_is_unchanged():
    rt, host, c = make()
    c("autobiography", "push", 7)
    with pytest.raises(ValueError):
        c("autobiography", "boom")
    assert c("autobiography", "total") == 7


def test_save_and_restore_continue_the_same_life(tmp_path):
    rt, host, c = make()
    for x in (1, 2, 3):
        c("autobiography", "push", x)
    snap = rt.save(SnapshotStore(tmp_path / "snaps"), label="t")
    c("autobiography", "push", 4)
    ref_digest, ref_tick = rt.store.digest(), rt.tick
    rt2, host2, c2 = make()
    rt2.restore(SnapshotStore(tmp_path / "snaps"), snap)
    c2("autobiography", "push", 4)
    assert rt2.store.digest() == ref_digest and rt2.tick == ref_tick
    assert c2("autobiography", "total") == 10


def test_projections_follow_the_stored_bytes_after_an_intervention():
    rt, host, c = make([(InterventionKind.ERASE, 1, 2, None)])
    c("autobiography", "push", 2); c("autobiography", "push", 3)        # erased after tick 1
    c("autobiography", "total")                                         # tick 2 reads the empty state
    proj = rt.store.read("autobiography.discrepancy")
    assert list(proj) == [-1.0, -1.0]                                    # neutral, not the stale [5, 2]


def test_driver_side_mechanism_calls_are_blocked_and_host_side_calls_run():
    rt, host, c = make()
    ran = []

    def legacy_fn(x):
        ran.append(x)
        return x * 2
    with pytest.raises(RouteBlocked):                                   # the driver cannot run it privately
        c.b.call("autobiography", "reliability_from_memory", legacy_fn, (3,), {})
    assert ran == [] and host.blocked == 1
    c.b.in_host = "autobiography"                                       # inside the owning host: frozen code runs
    try:
        assert c.b.call("autobiography", "reliability_from_memory", legacy_fn, (3,), {}) == 6
    finally:
        c.b.in_host = None
    assert ran == [3] and c.b.nested["autobiography->autobiography:reliability_from_memory"] == 1
