"""Tests for the store-mediated composition layer (v0.38.1.dev6, ENGINEERING_ONLY)."""

import math
import struct
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_brain_unified.channels import ChannelDecl, ChannelStore, ChannelWriteError, ChannelError
from continuous_brain_unified.composition import codec
from continuous_brain_unified.composition.boundary import (Boundary, Census, MechanismHost, RouteBlocked,
                                                          install, request_channel)
from continuous_brain_unified.composition.spec import (API, SPECS, CanonicalChannel, MechanismSpec,
                                                      SharedConstructor, spec_digest)
from continuous_brain_unified.runtime import BindingPolicy, RuntimeConfig, UnifiedRuntime
from continuous_brain_unified.slots import DRIVER_OWNER, SLOTS_BY_NAME


# ---------------------------------------------------------------- a fake legacy module (allowed by name prefix)
MOD = "evaluate_cbtest_fake_mech"


def _make_fake_module():
    """Fresh legacy-like objects per test, so no redirect can leak between tests."""
    m = types.ModuleType(MOD)
    src = """
from dataclasses import dataclass

@dataclass
class Phase:
    baseline: float
    strategy: int
    graduated: bool = False

class Gen:
    def __init__(self, gain=0.12):
        self.gain = gain
    def generate(self, rel, *, cycle):
        a, b = float(rel[0]), float(rel[1])
        if a == b:
            return None
        return Phase(min(a, b), 1 if a < b else 2)

def add(t, x):
    return t + x

def bump(phase):
    phase.strategy += 1
    phase.graduated = True
    return phase

def outer(t):
    return add(t, 1.0)

def boom():
    raise ValueError("legacy failure", 7)

def leak(lst):
    return {"inner": lst}

class Box:
    def __init__(self):
        self.items = []
        self.count = 0

def push_both(box, items):
    items.append(1)
    box.count += 1

def ret_kw(x, *, box):
    box.count += 5
    return box

def read_after_write(base, view):
    base[0] = 2.0
    return float(view[0])

def call_planner(x):
    return plan_it(x)

def plan_it(x):
    return x * 2

import torch as _torch

def load_net(path, device="cpu"):
    _torch.manual_seed(1234)
    return _torch.nn.Linear(3, 2), {"path": str(path)}

def use_net(net, x):
    return net(x).detach()
"""
    m.__dict__["__name__"] = MOD
    sys.modules[MOD] = m
    exec(compile(src, MOD, "exec"), m.__dict__)
    return m


@pytest.fixture
def fake_module():
    m = _make_fake_module()
    yield m
    for b in _BOUNDARIES:
        b.uninstall()
    _BOUNDARIES.clear()
    del sys.modules[MOD]


_BOUNDARIES: list = []


from continuous_brain_unified.composition.spec import SharedLoader

FAKE_SPECS = (
    MechanismSpec("fake_goal", "goal_genesis",
                  apis=(API(MOD, "Gen.generate"),),
                  constructors=(SharedConstructor(MOD, "Gen", "gen"),),
                  canonical=(CanonicalChannel("goal_genesis.active_goal", (4,)),)),
    MechanismSpec("fake_self", "self_state",
                  apis=(API(MOD, "add"), API(MOD, "bump"), API(MOD, "outer"), API(MOD, "boom"), API(MOD, "leak"),
                        API(MOD, "push_both"), API(MOD, "ret_kw"), API(MOD, "read_after_write"),
                        API(MOD, "call_planner"), API(MOD, "use_net")),
                  loaders=(SharedLoader(MOD, "load_net", ("net.pt",), "net"),),
                  canonical=(CanonicalChannel("self_state.vector", (6,)),)),
    MechanismSpec("fake_plan", "planner",
                  apis=(API(MOD, "plan_it"),),
                  canonical=(CanonicalChannel("planner.plan", (2,)),)),
)


def _runtime(boundary, hosts):
    driver = tuple(ChannelDecl(request_channel(s), DRIVER_OWNER, None, "uint8") for s in sorted(hosts))
    rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES, driver=driver))
    rt.bind_many(list(hosts.values()))
    boundary.attach(rt, hosts)
    return rt


def _install(fake_module, bound):
    b = Boundary(); census = Census()
    _BOUNDARIES.append(b)
    hosts = install(b, FAKE_SPECS, set(bound), Path.cwd(), census)
    return b, census, hosts, _runtime(b, hosts)


# ---------------------------------------------------------------- codec
def test_codec_roundtrip_is_bit_exact():
    t32 = torch.tensor([1.5, -0.0, float("inf"), 3.0e-40], dtype=torch.float32)
    vals = [None, True, False, 0, -2**70, 1.0, -0.0, float("nan"), 1e-310, "héllo", b"\x00\xff",
            (1, [2, 3]), {"b": 1, "a": 2}, range(3, 9, 2), torch.Size([2, 3]),
            t32, torch.zeros(0, 3), torch.tensor(5), torch.tensor([True, False]), torch.arange(6).reshape(2, 3).t(),
            np.array([[1, 2]], dtype=np.int16), np.float32(2.5), 3 + 4j]
    for v in vals:
        out = codec.decode(codec.encode(v).data)
        assert codec.encode(out).data == codec.encode(v).data
        assert type(out) is type(v)
    x = codec.decode(codec.encode(float("nan")).data)
    assert struct.pack("<d", x) == struct.pack("<d", float("nan"))
    assert list(codec.decode(codec.encode({"b": 1, "a": 2}).data)) == ["b", "a"]    # insertion order kept
    g = torch.ones(3, requires_grad=True)
    assert codec.decode(codec.encode(g).data).requires_grad


def test_codec_preserves_aliasing_and_fails_closed(fake_module):
    Phase = fake_module.Phase
    shared = [1, 2]
    out = codec.decode(codec.encode({"x": shared, "y": shared}).data)
    assert out["x"] is out["y"]
    p = Phase(0.3, 1)
    q = codec.decode(codec.encode([p, p]).data)
    assert q[0] is q[1] and q[0] == p and type(q[0]) is Phase
    with pytest.raises(codec.CodecError):
        codec.encode(object())
    with pytest.raises(codec.CodecError):
        codec.encode(torch.nn.Linear(2, 2))
    with pytest.raises(codec.CodecError):
        codec.encode(np.array([object()], dtype=object))
    with pytest.raises(codec.CodecError):
        codec.decode(b"nope")


def test_instance_tokens():
    table = codec.InstanceTable()
    lin = torch.nn.Linear(2, 2)
    table.register("m:shared", lin)
    out = codec.decode(codec.encode((lin, 3), table).data, table)
    assert out[0] is lin
    with pytest.raises(codec.CodecError):
        table.register("m:shared", torch.nn.Linear(2, 2))


# ---------------------------------------------------------------- channel store and runtime
def test_blob_channels_and_read_log():
    s = ChannelStore()
    with pytest.raises(ChannelError):
        s.declare(ChannelDecl("x.b", "x", None, "float64"))
    s.declare(ChannelDecl("x.b", "x", None, "uint8"))
    s.begin_tick(0)
    assert s.read("x.b", reader="r").size == 0
    s.write("x.b", b"abc", "x")
    assert bytes(s.read("x.b", reader="r")) == b"abc"
    with pytest.raises(ChannelWriteError):
        s.begin_tick(1); s.write("x.b", np.zeros((2, 2), dtype=np.uint8), "x")
    rep = {(r["reader"], r["channel"]): r for r in s.read_report()}
    assert rep[("r", "x.b")]["reads"] == 2 and rep[("r", "x.b")]["nonempty_reads"] == 1


def test_driver_channels_are_mandatory_each_tick():
    d = (ChannelDecl("driver.q", DRIVER_OWNER, None, "uint8"),)
    rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES, driver=d))
    with pytest.raises(ValueError):
        rt.tick_once()
    res = rt.tick_once(driver={"driver.q": b"hi"})
    assert "driver.q" in res.writes
    assert RuntimeConfig().config_hash() != RuntimeConfig(driver=d).config_hash()
    with pytest.raises(Exception):
        UnifiedRuntime(RuntimeConfig(driver=(ChannelDecl("self_state.q", "self_state", None, "uint8"),)))


# ---------------------------------------------------------------- boundary behaviour
def test_calls_cross_the_store_and_match_direct_execution(fake_module):
    Gen, Phase = fake_module.Gen, fake_module.Phase
    direct_gen = Gen()
    direct = direct_gen.generate(torch.tensor([0.2, 0.7]), cycle=0)
    b, census, hosts, rt = _install(fake_module, {"fake_goal", "fake_self"})
    g = fake_module.Gen()                      # every constructed instance is registered (stateless policy)
    assert b.instances.key_of(g) == "gen:0"
    assert census.constructed_instances[-1]["default_config"] is True
    got = g.generate(torch.tensor([0.2, 0.7]), cycle=0)
    assert got == direct and got is not direct
    t = torch.tensor([1.0, 2.0])
    assert torch.equal(fake_module.add(t, 2.0), t + 2.0)
    assert rt.tick == 1 and hosts["goal_genesis"].stats.requests == 1 and hosts["self_state"].stats.requests == 1
    p = Phase(0.1, 1)
    r = fake_module.bump(p)
    assert r is p and p.strategy == 2 and p.graduated          # in-place post-state and identity preserved
    assert hosts["self_state"].stats.mutated_args == 1
    assert torch.equal(fake_module.outer(t), t + 1.0)
    assert b.nested["self_state->self_state:add"] == 1
    with pytest.raises(ValueError) as e:
        fake_module.boom()
    assert e.value.args == ("legacy failure", 7)
    with pytest.raises(RouteBlocked):
        fake_module.leak([1])
    with pytest.raises(RouteBlocked):
        fake_module.add(object(), 1)
    reads = {(r["reader"], r["channel"]): r for r in rt.store.read_report()}
    assert reads[("goal_genesis", request_channel("goal_genesis"))]["nonempty_reads"] == 1
    assert reads[(DRIVER_OWNER, "goal_genesis.response")]["nonempty_reads"] == 1


def test_flows_between_hosts_are_attested(fake_module):
    b, census, hosts, rt = _install(fake_module, {"fake_goal", "fake_self"})
    rel = fake_module.add(torch.tensor([0.2, 0.7]), 0.0)       # self_state emits a tensor ...
    fake_module.Gen().generate(rel, cycle=0)                   # ... that goal_genesis then consumes
    assert b.content_flows["self_state->goal_genesis"] == 1
    fake_module.add(torch.zeros(3), 0.0)                       # uninformative content is never traced
    fake_module.Gen().generate(torch.zeros(2), cycle=0)
    assert b.content_flows["self_state->goal_genesis"] == 1


def test_unbound_mechanism_calls_are_blocked_not_run_privately(fake_module):
    b, census, hosts, rt = _install(fake_module, {"fake_self"})
    g = fake_module.Gen()                                      # unbound: an ordinary private object
    assert b.instances.key_of(g) is None
    with pytest.raises(RouteBlocked):
        g.generate(torch.tensor([0.2, 0.7]), cycle=0)
    assert b.unbound_calls["goal_genesis:Gen.generate"] == 1


def test_uninstall_restores_every_binding(fake_module):
    real_load, add0, gen0, init0 = torch.load, fake_module.add, fake_module.Gen.generate, fake_module.Gen.__init__
    b, census, hosts, rt = _install(fake_module, {"fake_goal", "fake_self"})
    assert torch.load is not real_load and fake_module.add is not add0 and fake_module.Gen.generate is not gen0
    b.uninstall(); _BOUNDARIES.remove(b)
    assert torch.load is real_load and fake_module.add is add0 and fake_module.Gen.generate is gen0
    assert fake_module.Gen.__init__ is init0 and "__new__" not in fake_module.Gen.__dict__
    assert fake_module.Gen() is not fake_module.Gen()


def test_six_real_hosts_declare_their_slots_canonical_channels():
    b = Boundary()
    hosts = {s.slot: MechanismHost(s, b) for s in SPECS}
    assert len(hosts) == 6
    for s in SPECS:
        declared = {d.name for d in hosts[s.slot].writes}
        assert set(SLOTS_BY_NAME[s.slot].canonical_writes) <= declared
    rt = _runtime(b, hosts)
    assert [c.slot for c in rt._order] == sorted([c.slot for c in rt._order],
                                                 key=[x.name for x in SLOTS_BY_NAME.values()].index)
    assert not rt.scientific
    assert len(spec_digest()) == 64


def test_store_override_reaches_the_caller(fake_module):
    """Store authority: what the caller gets is what the store holds after interventions."""
    from continuous_brain_protocols import InterventionKind, InterventionSpec
    from continuous_brain_unified.composition.codec import encode as enc
    b = Boundary(); census = Census(); _BOUNDARIES.append(b)
    hosts = install(b, FAKE_SPECS, {"fake_goal"}, Path.cwd(), census)
    driver = tuple(ChannelDecl(request_channel(s), DRIVER_OWNER, None, "uint8") for s in sorted(hosts))
    rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES, driver=driver))
    rt.bind_many(list(hosts.values()))
    b.attach(rt, hosts)
    donor = fake_module.Phase(0.99, 2)
    payload = enc({"status": "ok", "result": donor, "result_ref": None, "post_args": {}, "post_kwargs": {}}).data
    rt.interventions.add_payload("donor", np.frombuffer(payload, dtype=np.uint8))
    rt.interventions.register(InterventionSpec(InterventionKind.REPLACE, "goal_genesis.response", 1,
                                               payload_ref="donor"))
    g = fake_module.Gen()
    first = g.generate(torch.tensor([0.2, 0.7]), cycle=0)      # tick 0: the host's own answer
    second = g.generate(torch.tensor([0.2, 0.7]), cycle=0)     # tick 1: the store holds the donor answer
    assert first == fake_module.Phase(float(torch.tensor(0.2)), 1)    # the host's real result
    assert second == donor and second != first
    assert len(rt.interventions.records) == 1


def test_post_state_keeps_nested_identities_and_kwarg_results(fake_module):
    b, census, hosts, rt = _install(fake_module, {"fake_self"})
    box = fake_module.Box(); items = box.items
    fake_module.push_both(box, box.items)
    assert box.items is items and items == [1] and box.count == 1
    r = fake_module.ret_kw(3, box=box)
    assert r is box and box.count == 6
    base = torch.zeros(4)
    with pytest.raises(RouteBlocked):                           # views between arguments are refused
        fake_module.read_after_write(base, base[:2])


def test_nested_call_into_unbound_mechanism_is_blocked(fake_module):
    b, census, hosts, rt = _install(fake_module, {"fake_self"})
    with pytest.raises(RouteBlocked):
        fake_module.call_planner(3)
    assert b.unbound_calls["planner:plan_it"] == 1
    b2, *_ = (b,)
    b.uninstall(); _BOUNDARIES.remove(b)
    b, census, hosts, rt = _install(fake_module, {"fake_self", "fake_plan"})
    assert fake_module.call_planner(3) == 6
    assert b.nested["self_state->planner:plan_it"] == 1


def test_shared_loader_acquisition_is_verified_and_reloads_are_private(fake_module):
    b, census, hosts, rt = _install(fake_module, {"fake_self"})
    net, meta = fake_module.load_net("net.pt")
    assert b.instances.key_of(net) == "net:shared"
    assert census.acquisition_verifications[-1]["bit_identical"] is True
    net2, _ = fake_module.load_net("net.pt")                   # a reload: a genuine fresh instance
    assert net2 is not net and census.private_instances[-1]["reason"] == "reload_probe"
    x = torch.ones(1, 3)
    assert torch.equal(fake_module.use_net(net, x), net(x).detach())
    assert torch.equal(fake_module.use_net(net2, x), net2(x).detach())
    host_fw = sum(v for k, v in b.forwards.items() if "|host:" in k)
    driver_fw = sum(v for k, v in b.forwards.items() if k.endswith("|driver"))
    assert host_fw == 2 and driver_fw == 2                     # direct driver calls are counted, not hidden


def test_copies_of_constructed_instances_are_independent(fake_module):
    import copy as _copy
    b, census, hosts, rt = _install(fake_module, {"fake_goal"})
    g = fake_module.Gen()
    h = _copy.deepcopy(g)
    assert h is not g and b.instances.key_of(h) is None
    assert g.generate(torch.tensor([0.2, 0.7]), cycle=0) == h.generate(torch.tensor([0.2, 0.7]), cycle=0)


def test_codec_refuses_lossy_values(fake_module):
    class LL(list):
        pass
    LL.__module__ = MOD
    fake_module.LL = LL
    bad = [np.zeros(2, dtype=[("a", "f4"), ("b", "i4")]), np.ma.masked_array([1, 2], mask=[0, 1]),
           LL([1, 2]), torch.ones(2, requires_grad=True) * 2, torch.view_as_real(torch.ones(2, dtype=torch.complex64).conj())
           if False else torch.ones(2, dtype=torch.complex64).conj()]
    for v in bad:
        with pytest.raises(codec.CodecError):
            codec.encode(v)
    for junk in (codec.MAGIC + b"P" + b"\x05" * 8, codec.MAGIC + b"X" + b"\x01" * 8, codec.MAGIC + b"Q"):
        with pytest.raises(codec.CodecError):
            codec.decode(junk)


def test_host_state_round_trips(fake_module):
    b, census, hosts, rt = _install(fake_module, {"fake_self"})
    h = hosts["self_state"]
    st = h.state_dict()
    h._held["self_state.vector"] = np.ones(6)
    h.load_state_dict(st)
    assert np.array_equal(h._held["self_state.vector"], np.full(6, -1.0))
