import numpy as np
import pytest

from continuous_brain_protocols import InterventionError, InterventionKind, InterventionSpec
from continuous_brain_snn.snapshot import SnapshotStore
from continuous_brain_unified import (REQUIRED_SLOTS, BindingError, ChannelDecl, ChannelStore, ChannelWriteError,
                                      Coprocessor, CoprocessorBudgetExceeded, CoprocessorResponse, RuntimeConfig,
                                      SchedulingError, UnifiedRuntime)

from fixtures import OBS, Accumulator, fixture_runtime, observation


def test_boots_with_every_mechanism_slot_unbound_and_says_why():
    rt = UnifiedRuntime()
    report = rt.binding_report()
    assert len(REQUIRED_SLOTS) == 15 and len(report) == 15
    mech = [r for r in report if r.kind == "mechanism"]
    infra = [r for r in report if r.kind == "infrastructure"]
    assert len(mech) == 12 and len(infra) == 3
    assert all(r.status == "unbound" and "DAV-74" in r.reason for r in mech)
    assert {r.slot for r in infra} == {"persistence", "provenance", "coprocessor"}
    assert rt.scientific                                    # nothing fake is bound
    rt.run(3)                                               # an empty individual still ticks
    assert rt.tick == 2


def test_test_fixtures_are_refused_by_the_default_policy():
    rt = UnifiedRuntime(RuntimeConfig(observation=OBS))
    with pytest.raises(BindingError, match="test fixture"):
        rt.bind(Accumulator("wm", "working_memory", ["working_memory.contents"], reads=["environment.joint_angle"]))
    assert not fixture_runtime().scientific


def test_binding_rules():
    rt = fixture_runtime(with_autobiography=False)
    with pytest.raises(BindingError, match="already bound"):
        rt.bind(Accumulator("wm2", "working_memory", ["working_memory.contents"]))
    with pytest.raises(BindingError, match="infrastructure"):
        rt.bind(Accumulator("p", "persistence", ["persistence.x"]))
    with pytest.raises(BindingError, match="canonical"):
        rt.bind(Accumulator("ep", "episodic_memory", ["episodic_memory.other"]))
    with pytest.raises(SchedulingError, match="nothing writes"):
        rt.bind(Accumulator("ep", "episodic_memory", ["episodic_memory.slots"], reads=["planner.plan"]))
    assert "episodic_memory.slots" not in rt.store.names()   # a failed bind leaves no declarations behind
    rt.run(1, lambda t: observation(0, t))
    with pytest.raises(BindingError, match="before the first tick"):
        rt.bind(Accumulator("ep", "episodic_memory", ["episodic_memory.slots"]))


def test_single_writer_namespace_and_read_only_values():
    s = ChannelStore()
    s.declare(ChannelDecl("planner.plan", "planner", (2,)))
    with pytest.raises(Exception, match="namespace"):
        s.declare(ChannelDecl("planner.other", "skills", (2,)))
    s.begin_tick(0)
    with pytest.raises(ChannelWriteError, match="may not write"):
        s.write("planner.plan", np.zeros(2), "skills")
    s.write("planner.plan", np.ones(2), "planner")
    with pytest.raises(ChannelWriteError, match="twice"):
        s.write("planner.plan", np.ones(2), "planner")
    with pytest.raises(ValueError):
        s.read("planner.plan")[0] = 3.0
    with pytest.raises(ChannelWriteError, match="shape"):
        s.begin_tick(1)
        s.write("planner.plan", np.ones(3), "planner")
    with pytest.raises(ChannelWriteError, match="intervention manager"):
        s.override("planner.plan", np.zeros(2), object(), "x")


def test_schedule_orders_writers_before_readers():
    rt = fixture_runtime()
    res = rt.tick_once(observation(0, 0))
    assert res.order.index("wm") < res.order.index("auto") < res.order.index("self")
    rt.tick_once(observation(0, 1))
    wm = rt.store.read("working_memory.contents")
    assert wm[0] == pytest.approx(observation(0, 1)["joint_angle"].sum() + 2)


def test_delayed_edges_allow_recurrence_and_same_tick_cycles_are_rejected():
    from continuous_brain_unified import BindingPolicy, schedule
    rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES))
    rt.bind_many([Accumulator("a", "planner", ["planner.plan"], delayed=["skills.selected_option"]),
                  Accumulator("b", "skills", ["skills.selected_option"], reads=["planner.plan"])])
    rt.run(2)
    # tick 1: planner sums the *previous* tick's option (2 elements x 3.0) and adds its count
    assert rt.store.read("skills.selected_option", delayed=True)[0] == 3.0
    assert rt.store.read("planner.plan")[0] == 2 * 3.0 + 2
    with pytest.raises(SchedulingError, match="cycle"):
        schedule([Accumulator("a", "planner", ["planner.plan"], reads=["skills.selected_option"]),
                  Accumulator("b", "skills", ["skills.selected_option"], reads=["planner.plan"])], set())
    with pytest.raises(SchedulingError, match="own output"):
        schedule([Accumulator("a", "planner", ["planner.plan"], reads=["planner.plan"])], set())


def test_interventions_erase_freeze_donor_replace_and_lock():
    rt = fixture_runtime()
    m = rt.interventions
    m.add_payload("donor", {t: np.full(2, 7.0 + t) for t in range(10)})
    m.add_payload("const", np.full(2, -1.0))
    m.register(InterventionSpec(InterventionKind.ERASE, "autobiography.record", 2, 4))
    m.register(InterventionSpec(InterventionKind.FREEZE, "working_memory.contents", 3, 6))
    m.register(InterventionSpec(InterventionKind.DONOR, "self_state.vector", 1, 3, "donor"))
    m.register(InterventionSpec(InterventionKind.REPLACE, "autobiography.discrepancy", 5, 6, "const"))
    with pytest.raises(InterventionError, match="overlapping"):
        m.register(InterventionSpec(InterventionKind.ERASE, "autobiography.record", 3, 5))
    seen = {}
    for t in range(7):
        rt.tick_once(observation(0, t))
        seen[t] = {c: np.array(rt.store.read(c)) for c in rt.store.names()}
    assert np.all(seen[2]["autobiography.record"] == 0) and np.all(seen[3]["autobiography.record"] == 0)
    assert not np.all(seen[4]["autobiography.record"] == 0)
    assert np.array_equal(seen[3]["working_memory.contents"], seen[2]["working_memory.contents"])
    assert np.array_equal(seen[5]["working_memory.contents"], seen[2]["working_memory.contents"])
    assert np.all(seen[1]["self_state.vector"] == 8.0) and np.all(seen[2]["self_state.vector"] == 9.0)
    assert np.all(seen[5]["autobiography.discrepancy"] == -1.0)
    # downstream readers saw the intervened value in the same tick: self_state reads the discrepancy
    assert seen[5]["self_state.vector"][0] == pytest.approx(-2.0 + rt.components["self_state"].count - 1)
    assert len(rt.view().intervention_records()) == 2 + 3 + 2 + 1
    with pytest.raises(InterventionError, match="locked"):
        m.register(InterventionSpec(InterventionKind.ERASE, "self_state.vector", 9))


def test_evaluator_view_is_read_only():
    rt = fixture_runtime()
    rt.run(2, lambda t: observation(0, t))
    v = rt.view()
    with pytest.raises(AttributeError):
        v.store = None
    assert not hasattr(v, "store") and not hasattr(v, "interventions")
    with pytest.raises(ValueError):
        v.read("working_memory.contents")[0] = 1.0
    assert len(v.write_log()) == 2 * 5


class Echo(Coprocessor):
    name = "echo"

    def query(self, request):
        return CoprocessorResponse(True, self.name, {"echo": dict(request.payload)})


def test_coprocessor_disabled_by_default_logged_budgeted_and_ablatable(tmp_path):
    rt = fixture_runtime(coprocessor=Echo())
    rt.run(3, lambda t: observation(0, t))
    assert rt.components["self_state"].replies == 0            # disabled at boot
    assert len(rt.coprocessor.log.entries) == 3 and all(not e["enabled"] for e in rt.coprocessor.log.entries)
    on = fixture_runtime(coprocessor=Echo(), enabled=True)
    on.run(3, lambda t: observation(0, t))
    assert on.components["self_state"].replies == 3
    assert all(e["request_sha256"] and e["response_sha256"] for e in on.coprocessor.log.entries)
    on.coprocessor.ablate()
    on.run(1, lambda t: observation(0, t))
    assert on.components["self_state"].replies == 3
    on.coprocessor.enabled = True
    on.coprocessor.begin_tick(99)
    on.coprocessor.query("x", "k", {})
    with pytest.raises(CoprocessorBudgetExceeded):
        on.coprocessor.query("x", "k", {})


def test_save_restore_is_exact_and_refuses_a_different_body(tmp_path):
    store = SnapshotStore(tmp_path)
    a = fixture_runtime(seed=1)
    a.run(4, lambda t: observation(1, t))
    snap = a.save(store)
    assert store.verify(snap)
    a.run(5, lambda t: observation(1, t))
    b = fixture_runtime(seed=1)
    b.restore(store, snap)
    assert b.tick == 3
    b.run(5, lambda t: observation(1, t))
    assert a.state_hash() == b.state_hash()
    c = fixture_runtime(seed=1, with_autobiography=False)
    with pytest.raises(BindingError, match="differs"):
        c.restore(store, snap)


def test_reads_cannot_be_made_writable_and_payloads_are_frozen():
    rt = fixture_runtime()
    pay = np.full(2, 5.0)
    rt.interventions.add_payload("p", pay)
    rt.interventions.register(InterventionSpec(InterventionKind.REPLACE, "autobiography.discrepancy", 1, 3, "p"))
    rt.tick_once(observation(0, 0))
    arr = rt.view().read("working_memory.contents")
    with pytest.raises(ValueError):
        arr.setflags(write=True)
    pay[:] = 99.0                                            # the caller's array is not the payload
    rt.tick_once(observation(0, 1))
    assert np.all(rt.store.read("autobiography.discrepancy") == 5.0)
    with pytest.raises(Exception):
        rt.interventions.specs.append(None)
    with pytest.raises(InterventionError):
        rt.interventions.payloads["q"] = np.zeros(2)


def test_components_get_a_query_only_coprocessor_handle():
    seen = {}

    class Grab(Accumulator):
        def step(self, ctx):
            seen["h"] = ctx.coprocessor
            return super().step(ctx)

    from continuous_brain_unified import BindingPolicy
    rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES))
    rt.bind(Grab("g", "planner", ["planner.plan"], uses_coprocessor=True))
    rt.tick_once()
    h = seen["h"]
    assert not hasattr(h, "enabled") and not hasattr(h, "provider")
    with pytest.raises(AttributeError):
        h.enabled = True
    assert not h.query("g", "k", {}).available


def test_failed_restore_leaves_the_runtime_untouched(tmp_path):
    store = SnapshotStore(tmp_path)
    a = fixture_runtime(seed=2)
    a.run(3, lambda t: observation(2, t))
    snap = a.save(store)
    b = fixture_runtime(seed=2)
    b.run(1, lambda t: observation(2, t))
    before = b.state_hash()
    import continuous_brain_unified.runtime as rmod
    orig = rmod.UnifiedRuntime.state_hash
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        return "corrupt" if calls["n"] == 1 else orig(self)
    try:
        rmod.UnifiedRuntime.state_hash = flaky
        with pytest.raises(BindingError):
            b.restore(store, snap)
    finally:
        rmod.UnifiedRuntime.state_hash = orig
    assert b.state_hash() == before and b.tick == 0
