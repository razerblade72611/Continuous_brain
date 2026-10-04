"""W1 world: determinism, snapshots, every obligation family (fulfil and fail), changes, daylight deadlines,
visibility ground truth, utterance grammar, the official Crafter score, and JSON-clean layer state."""
import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from crafter import constants, objects

from continuous_brain_world import W1Config, W1Env, aggregate, crafter_score
from continuous_brain_world import daylight as dl
from continuous_brain_world.baselines import RandomAgent, ScriptedAgent
from continuous_brain_world.caretaker import choose_deadline, dark_steps
from continuous_brain_world.perception import perceivable_grid, perceived_now
from continuous_brain_world.utterance import compass, distance_bucket, format_offset, parse, parse_payload

REPO = Path(__file__).resolve().parents[1]
A = {name: i for i, name in enumerate(constants.actions)}


# --- helpers ---------------------------------------------------------------------------------------------
def quiet(**kw):
    """A config whose caretaker never speaks on its own (scenarios issue obligations explicitly)."""
    base = dict(episode_length=2000, arrival_rate0=0.0, arrival_rate_max=0.0, change_rate=0.0)
    base.update(kw)
    return W1Config(**base)


def make_env(seed=7, **kw):
    env = W1Env(quiet(**kw))
    env.reset(seed)
    return env


def step(env, action="noop", utter=None):
    return env.step(A[action], utter)


def player(env):
    return env.bridge.player


def world(env):
    return env.bridge.world


def clear_creatures(env):
    for obj in list(world(env).objects):
        if not isinstance(obj, objects.Player):
            world(env).remove(obj)


def teleport(env, pos):
    w, p = world(env), player(env)
    pos = (int(pos[0]), int(pos[1]))
    if w[pos][1] is not None and w[pos][1] is not p:
        w.remove(w[pos][1])
    w[pos] = "grass"
    w.move(p, pos)


def face(env, direction, material="grass"):
    """Make the player face ``direction`` with ``material`` in front of it (no step is taken)."""
    p = player(env)
    p.facing = direction
    world(env)[(p.pos[0] + direction[0], p.pos[1] + direction[1])] = material


def keep_alive(env, action):
    """Refill the needs and veto walking into lava, so long random runs reach nights, spawns and despawns."""
    p = env.bridge.player
    for k in ("health", "food", "drink", "energy"):
        p.inventory[k] = 9
    name = constants.actions[int(action)]
    if name.startswith("move_"):
        d = {"move_left": (-1, 0), "move_right": (1, 0), "move_up": (0, -1), "move_down": (0, 1)}[name]
        if env.bridge.world[(p.pos[0] + d[0], p.pos[1] + d[1])][0] == "lava":
            return 0
    return int(action)


def run_signature(env, seed, n, rng_seed=0, utter_every=37):
    """Run n steps with seeded random actions/utterances; return a digest of everything observable."""
    obs = env.reset(seed)
    rng = np.random.default_rng(rng_seed)
    h = hashlib.sha256(obs["image"].tobytes())
    for i in range(n):
        utter = f"answer q{1 + i % 5}: {i % 4} east" if i % utter_every == 0 else None
        obs, r, d, info = env.step(keep_alive(env, rng.integers(17)), utter)
        h.update(obs["image"].tobytes())
        h.update(json.dumps([obs["messages"], obs["step"], obs["phase"], r], sort_keys=True).encode())
        if d:
            break
    h.update(json.dumps(env.layer_state(), sort_keys=True).encode())
    return h.hexdigest()


# --- determinism -----------------------------------------------------------------------------------------
def test_same_seed_same_everything_and_different_seed_differs():
    cfg = dict(episode_length=600, arrival_rate0=1 / 40, change_rate=1 / 150)
    a = run_signature(W1Env(W1Config(**cfg)), 11, 320)
    b = run_signature(W1Env(W1Config(**cfg)), 11, 320)
    c = run_signature(W1Env(W1Config(**cfg)), 12, 320)
    assert a == b
    assert a != c


def test_determinism_across_processes():
    """Vanilla Crafter diverges across processes (its despawn picks from a set ordered by memory address);
    W1's fix must make two processes with different memory layouts agree."""
    code = textwrap.dedent(f"""
        import sys, os
        sys.path.insert(0, {str(REPO)!r})
        pad = [object() for _ in range(int(os.environ["W1_PAD"]))]   # perturb object addresses
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from test_w1_world import run_signature
        from continuous_brain_world import W1Env, W1Config
        print(run_signature(W1Env(W1Config(episode_length=700, arrival_rate0=1/40, change_rate=1/150)), 5, 420))
    """)
    outs = []
    for pad, hseed in (("0", "1"), ("4321", "999")):
        env = dict(os.environ, W1_PAD=pad, PYTHONHASHSEED=hseed)
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
        assert res.returncode == 0, res.stderr
        outs.append(res.stdout.strip())
    assert outs[0] == outs[1]


def test_large_render_has_no_side_effect_on_dynamics():
    def images(cfg):
        env = W1Env(cfg)
        env.reset(3)
        rng = np.random.default_rng(1)
        out = []
        for _ in range(230):  # into the night, where rendering draws noise from the world RNG
            obs, *_ = env.step(keep_alive(env, rng.integers(17)))
            out.append(obs["image"].copy())
            if "image_large" in obs:
                assert obs["image_large"].shape == (128, 128, 3) and obs["image_large"].dtype == np.uint8
        return out

    plain = images(W1Config(episode_length=500))
    large = images(W1Config(episode_length=500, large_render_size=128))
    assert all(np.array_equal(x, y) for x, y in zip(plain, large))
    assert plain[0].shape == (64, 64, 3) and plain[0].dtype == np.uint8


# --- snapshot / restore ----------------------------------------------------------------------------------
def test_snapshot_restore_is_bit_exact_and_survives_json():
    cfg = W1Config(episode_length=900, arrival_rate0=1 / 30, change_rate=1 / 120)
    env = W1Env(cfg)
    env.reset(21)
    rng = np.random.default_rng(4)
    actions = [int(x) for x in rng.integers(17, size=420)]
    for a in actions[:220]:
        env.step(keep_alive(env, a))
    snap = env.snapshot()
    text = W1Env.snapshot_to_json(snap)

    def tail(e):
        rec = []
        for i, a in enumerate(actions[220:]):
            obs, r, d, info = e.step(keep_alive(e, a), "answer q3: never" if i == 7 else None)
            rec.append((obs["image"].tobytes(), json.dumps([obs["messages"], r, d, info["events"]], sort_keys=True,
                                                           default=str)))
        return rec, json.dumps(e.layer_state(), sort_keys=True)

    first = tail(env)
    env.restore(snap)
    second = tail(env)
    fresh = W1Env(cfg)
    fresh.restore(W1Env.snapshot_from_json(text))
    third = tail(fresh)
    assert first == second == third
    with pytest.raises(ValueError):
        W1Env(W1Config(episode_length=901)).restore(snap)


def test_layer_state_is_json_serialisable_after_a_busy_life():
    env = W1Env(W1Config(episode_length=700, arrival_rate0=1 / 25, change_rate=1 / 100))
    obs = env.reset(2)
    agent = ScriptedAgent(2)
    info = env.privileged_now()
    for _ in range(650):
        a, u = agent.act(obs, info)
        obs, r, d, info = env.step(a, u)
        if d:
            break
    state = env.layer_state()
    text = json.dumps(state, sort_keys=True, allow_nan=False)
    assert json.loads(text) == state
    fams = {o["family"] for o in state["obligations"]}
    assert len(fams) >= 5, fams
    assert any(h["event"] in ("amended", "cancelled") for o in state["obligations"] for h in o["history"])


# --- obligation families ---------------------------------------------------------------------------------
def test_collect_fulfilled_by_net_gain_and_failed_at_deadline():
    env = make_env()
    p = player(env)
    p.inventory["wood"] = 1
    ok = env.issue("collect", {"item": "wood", "quantity": 2, "base": 1, "target": 3}, deadline=60)
    late = env.issue("collect", {"item": "stone", "quantity": 1, "base": 0, "target": 1}, deadline=5)
    obs, *_ = step(env)
    assert ok["text"] in obs["messages"] and late["text"] in obs["messages"]
    p.inventory["wood"] = 2
    step(env)
    assert ok["status"] == "open"
    face(env, (1, 0), "tree")
    obs, r, d, info = step(env, "do")
    assert p.inventory["wood"] == 3 and ok["status"] == "fulfilled" and ok["resolved_step"] == env.bridge.step_count
    assert r["obligations"] == pytest.approx(ok["value"])
    while env.bridge.step_count < 5:
        step(env)
    assert late["status"] == "failed" and late["resolved_step"] == 5 and late["reason"] == "deadline"


def test_craft_needs_a_new_item_made_after_the_request():
    env = make_env()
    p = player(env)
    p.inventory["wood"] = 3
    p.inventory["wood_pickaxe"] = 1
    ob = env.issue("craft", {"tool": "wood_pickaxe", "base": p.achievements["make_wood_pickaxe"]}, deadline=40)
    step(env)
    assert ob["status"] == "open"  # already owning one does not count
    face(env, (1, 0), "table")
    step(env, "make_wood_pickaxe")
    assert ob["status"] == "fulfilled" and p.inventory["wood_pickaxe"] == 2
    miss = env.issue("craft", {"tool": "stone_sword", "base": 0}, deadline=env.bridge.step_count + 3)
    for _ in range(3):
        step(env)
    assert miss["status"] == "failed" and miss["reason"] == "deadline"


def test_place_near_and_onto_rules():
    env = make_env()
    clear_creatures(env)
    p = player(env)
    p.inventory["wood"] = 9
    p.inventory["stone"] = 3
    x, y = int(p.pos[0]), int(p.pos[1])
    near = env.issue("place", {"what": "table", "near": "water", "radius": 4, "base": 0}, deadline=200)
    onto = env.issue("place", {"what": "stone", "onto": "water", "base": 0}, deadline=200)
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            world(env)[(x + dx, y + dy)] = "grass"
    step(env)
    face(env, (1, 0))
    step(env, "place_table")  # no water anywhere near: placed, but the request is not met
    assert world(env)[(x + 1, y)][0] == "table" and near["status"] == "open"
    face(env, (0, 1))
    world(env)[(x + 3, y + 1)] = "water"   # Chebyshev distance 3 from the next table tile, (x, y + 1)
    step(env, "place_table")
    assert near["status"] == "fulfilled"
    face(env, (-1, 0))
    step(env, "place_stone")              # onto grass: not "onto water"
    assert onto["status"] == "open"
    face(env, (0, -1), "water")
    step(env, "place_stone")
    assert onto["status"] == "fulfilled" and world(env)[(x, y - 1)][0] == "stone"
    plant = env.issue("place", {"what": "plant", "near": "table", "radius": 1, "base": 0},
                      deadline=env.bridge.step_count + 2)
    step(env)
    step(env)
    assert plant["status"] == "failed"


def test_maintain_kept_and_violated():
    env = make_env()
    p = player(env)
    kept = env.issue("maintain", {"stat": "drink", "threshold": 3}, deadline=12)
    broken = env.issue("maintain", {"stat": "food", "threshold": 6}, deadline=50)
    for s in range(1, 12):
        p.inventory["drink"] = 5
        p.inventory["food"] = 8 if s < 6 else 5
        step(env)
        if s < 6:
            assert broken["status"] == "open"
    assert broken["status"] == "failed" and broken["reason"] == "violated" and broken["resolved_step"] == 6
    assert kept["status"] == "open"
    step(env)
    assert kept["status"] == "fulfilled" and kept["resolved_step"] == 12


def test_avoid_is_source_specific():
    env = make_env()
    clear_creatures(env)
    p = player(env)
    zombie = env.issue("avoid", {"source": "zombie"}, deadline=100)
    anyd = env.issue("avoid", {"source": "any"}, deadline=100)
    arrow = env.issue("avoid", {"source": "arrow"}, deadline=100)
    calm = env.issue("avoid", {"source": "zombie"}, deadline=3)
    step(env)
    x, y = int(p.pos[0]), int(p.pos[1])
    world(env)[(x + 1, y)] = "grass"
    world(env).add(objects.Arrow(world(env), (x + 1, y), (-1, 0)))
    obs, r, d, info = step(env)
    assert info["events"]["arrow_hits"] == 1 and info["events"]["health_drop"]
    assert arrow["status"] == "failed" and anyd["status"] == "failed" and zombie["status"] == "open"
    step(env)
    assert calm["status"] == "fulfilled" and calm["reason"] == "kept"
    world(env)[(x, y + 1)] = "grass"
    world(env).add(objects.Zombie(world(env), (x, y + 1), p))
    for _ in range(10):
        obs, r, d, info = step(env)
        if info["events"]["zombie_hits"]:
            break
    assert info["events"]["zombie_hits"] == 1
    assert zombie["status"] == "failed" and zombie["resolved_step"] == env.bridge.step_count


def test_visit_needs_the_stated_direction_then_a_return():
    env = make_env()
    clear_creatures(env)
    p = player(env)
    start = (int(p.pos[0]), int(p.pos[1]))
    for dx in range(-14, 15):
        for dy in range(-14, 15):
            world(env)[(start[0] + dx, start[1] + dy)] = "grass"
    world(env)[(start[0] + 12, start[1] + 1)] = "water"     # east
    world(env)[(start[0], start[1] + 12)] = "water"         # south: wrong direction
    spec = {"feature": "water", "direction": "east", "start": list(start), "min_dist": 8, "reach_radius": 3,
            "return_radius": 3}
    ob = env.issue("visit", spec, deadline=80)
    fail = env.issue("visit", dict(spec, direction="north"), deadline=6)
    step(env)
    teleport(env, (start[0], start[1] + 10))
    step(env)
    assert ob["progress"]["reached_step"] is None
    teleport(env, (start[0] + 10, start[1]))
    step(env)
    assert ob["progress"]["reached_step"] == env.bridge.step_count and ob["status"] == "open"
    teleport(env, (start[0] + 2, start[1] - 1))
    step(env)
    assert ob["status"] == "fulfilled" and ob["reason"] == "returned"
    while env.bridge.step_count < 6:
        step(env)
    assert fail["status"] == "failed" and fail["reason"] == "deadline"


def test_report_questions_graded_against_the_visibility_log():
    env = make_env()
    clear_creatures(env)
    p = player(env)
    x, y = int(p.pos[0]), int(p.pos[1])
    world(env)[(x + 3, y - 2)] = "grass"
    cow = objects.Cow(world(env), (x + 3, y - 2))
    world(env).add(cow)
    step(env)
    seen = env.layer.state["visibility"]["last_seen"]["cow"]
    cow_pos = seen["positions"][0]
    world(env).remove(cow)
    step(env)
    q_ok = env.issue("report", {"question": "last_seen", "kind": "cow"}, deadline=30)
    q_bucket = env.issue("report", {"question": "last_seen", "kind": "cow"}, deadline=30)
    q_wrong = env.issue("report", {"question": "last_seen", "kind": "cow"}, deadline=30)
    q_never = env.issue("report", {"question": "last_seen", "kind": "diamond"}, deadline=30)
    q_silent = env.issue("report", {"question": "last_seen", "kind": "cow"}, deadline=env.bridge.step_count + 3)
    step(env)
    dx, dy = cow_pos[0] - int(p.pos[0]), cow_pos[1] - int(p.pos[1])
    bucket = f"{compass(dx, dy)} {distance_bucket(dx, dy)}"
    utter = (f"answer {q_ok['id']}: {format_offset(dx, dy)}; answer {q_bucket['id']}: {bucket}; "
             f"answer {q_wrong['id']}: {format_offset(-dx - 6, dy)}; answer {q_never['id']}: never; ack o1")
    obs, r, d, info = step(env, "noop", utter)
    assert q_ok["status"] == "fulfilled" and q_bucket["status"] == "fulfilled" and q_never["status"] == "fulfilled"
    assert q_wrong["status"] == "failed" and q_wrong["reason"] == "wrong"
    assert any("The answer was" in m for m in obs["messages"])
    step(env, "noop", f"answer {q_ok['id']}: 0 east")  # a second answer to a graded question is ignored
    assert q_ok["status"] == "fulfilled" and env.layer.state["stats"]["answers_ignored"] == 1
    while q_silent["status"] == "open":
        step(env)
    assert q_silent["status"] == "failed" and q_silent["reason"] == "no_answer"


def test_count_and_request_memory_questions():
    env = make_env()
    p = player(env)
    task = env.issue("collect", {"item": "wood", "quantity": 1, "base": 0, "target": 1}, deadline=500,
                     keywords=["wood", "logs"])
    step(env)
    face(env, (1, 0), "tree")
    step(env, "do")
    face(env, (1, 0), "tree")
    step(env, "do")
    s = env.bridge.step_count
    count = env.issue("report", {"question": "count_today", "event": "collect_wood", "day": 0}, deadline=s + 20)
    first = env.issue("report", {"question": "first_request", "window": [0, s]}, deadline=s + 20)
    first_kw = env.issue("report", {"question": "first_request", "window": [0, s]}, deadline=s + 20)
    nreq = env.issue("report", {"question": "count_requests", "window": [0, s]}, deadline=s + 20)
    none = env.issue("report", {"question": "last_cancelled"}, deadline=s + 20)
    step(env)
    step(env, "noop", f"answer {count['id']}: two; answer {first['id']}: {task['id']}; "
                      f"answer {first_kw['id']}: the logs; answer {nreq['id']}: 1; answer {none['id']}: nothing")
    assert p.achievements["collect_wood"] == 2
    for q in (count, first, first_kw, nreq, none):
        assert q["status"] == "fulfilled", (q["spec"], q.get("answer"))


def test_amendment_and_cancellation_semantics():
    env = make_env()
    p = player(env)
    ob = env.issue("collect", {"item": "stone", "quantity": 1, "base": 0, "target": 1}, deadline=50,
                   keywords=["stone"])
    gone = env.issue("collect", {"item": "wood", "quantity": 1, "base": 0, "target": 1}, deadline=50,
                     keywords=["wood"])
    step(env)
    msg = env.change(ob["id"], "amend", "Change of plan: I need 3 stones.", {"spec": {"quantity": 3, "target": 3},
                                                                               "deadline": 80})
    env.change(gone["id"], "cancel", "Never mind the wood.")
    obs, *_ = step(env)
    assert msg in obs["messages"] and f"[{gone['id']}] Never mind the wood." in obs["messages"]
    assert ob["status"] == "amended" and ob["deadline"] == 80
    assert ob["history"][-1]["before"] == {"spec": {"item": "stone", "quantity": 1, "base": 0, "target": 1},
                                           "deadline": 50}
    p.inventory["stone"] = 1
    p.inventory["wood"] = 5     # meets the cancelled request: it must stay cancelled
    step(env)
    assert ob["status"] == "amended" and gone["status"] == "cancelled"
    p.inventory["stone"] = 3
    step(env)
    assert ob["status"] == "fulfilled"
    summary = env.episode_summary()
    fam = summary["obligations"]["per_family"]["collect"]
    assert fam["fulfilled"] == 1 and fam["cancelled"] == 1 and fam["amended"] == 1 and fam["on_time_rate"] == 1.0
    with pytest.raises(ValueError):
        env.change(gone["id"], "amend", "too late", {})


def test_caretaker_issues_amends_and_cancels_on_its_own():
    env = W1Env(W1Config(episode_length=1500, arrival_rate0=1 / 15, change_rate=1 / 40))
    obs = env.reset(9)
    rng = np.random.default_rng(0)
    texts = list(obs["messages"])
    for _ in range(900):
        obs, r, d, info = env.step(keep_alive(env, rng.integers(17)))
        texts += obs["messages"]
    obs_ = env.layer.obligations
    assert len({o["family"] for o in obs_}) == 7
    assert sum(o["status"] == "cancelled" for o in obs_) >= 1
    assert sum(any(h["event"] == "amended" for h in o["history"]) for o in obs_) >= 1
    assert max(sum(1 for o in obs_ if o["issue_step"] <= s and (o["resolved_step"] is None or o["resolved_step"] > s))
               for s in range(0, 900)) <= 5
    assert all(m.startswith("[") or m.startswith("Good morning") for m in texts)


def test_death_fails_everything_active():
    env = make_env()
    ob = env.issue("collect", {"item": "wood", "quantity": 1, "base": 0, "target": 1}, deadline=500)
    step(env)
    player(env).health = 0
    obs, r, d, info = step(env)
    assert d and ob["status"] == "failed" and ob["reason"] == "died"
    assert env.layer.state["death_step"] == env.bridge.step_count
    with pytest.raises(RuntimeError):
        step(env)


# --- daylight arithmetic -----------------------------------------------------------------------------------
def test_daylight_replica_and_cycle_constants():
    assert (dl.NIGHTFALL_T, dl.DAWN_T, dl.NOON_T, dl.MIDNIGHT_T) == (148, 273, 60, 210)
    assert dl._verify_periodicity(30000)
    env = W1Env(W1Config(episode_length=800))
    env.reset(1)
    for s in range(1, 650):
        obs, *_ = env.step(keep_alive(env, 0))
        assert world(env).daylight == dl.daylight(s)
        assert obs["phase"] == dl.phase(s) and obs["day"] == dl.day_index(s)
    assert dl.day_index(272) == 0 and dl.day_index(273) == 1 and dl.day_index(572) == 1 and dl.day_index(573) == 2
    assert dl.phase(59) == "morning" and dl.phase(60) == "afternoon" and dl.phase(148) == "night"
    assert dl.phase(273) == "morning" and dl.morning_window(1) == (273, 360)


def test_next_events_match_brute_force():
    rng = np.random.default_rng(3)
    for s in [0, 1, 147, 148, 272, 273, 299, 300] + list(rng.integers(0, 5000, 60)):
        s = int(s)
        nf = next(t for t in range(s + 1, s + 700) if dl.is_dark(t) and not dl.is_dark(t - 1))
        dw = next(t for t in range(s + 1, s + 700) if not dl.is_dark(t) and dl.is_dark(t - 1))
        nn = next(t for t in range(s + 1, s + 700) if t % 300 == dl.NOON_T)
        assert (dl.next_nightfall(s), dl.next_dawn(s), dl.next_noon(s)) == (nf, dw, nn)
        assert dark_steps(s, s + 777) == sum(dl.is_dark(t) for t in range(s, s + 777))


def test_deadline_phrases_mean_what_they_say():
    kinds = set()
    rng = np.random.default_rng(0)
    for i in range(1500):
        s = int(rng.integers(0, 3000))
        until = bool(i % 2)
        d, phrase = choose_deadline(rng, s, float(rng.integers(40, 400)), until=until)
        if phrase in ("before nightfall", "before it gets dark", "until nightfall"):
            kind = "nightfall"
            assert d == dl.next_nightfall(s) - 1 and not dl.is_dark(d) and dl.is_dark(d + 1)
        elif phrase in ("by tomorrow morning", "by morning", "until morning", "through the night until morning"):
            kind = "dawn"
            assert d == dl.next_dawn(s) and not dl.is_dark(d) and dl.is_dark(d - 1)
        elif phrase == "by noon":
            kind = "noon"
            assert d == dl.next_noon(s) and d % 300 == 60
        elif phrase.startswith(("by step ", "until step ")):
            kind = "step"
            assert d == int(phrase.rsplit(" ", 1)[1])
        else:
            kind = "within"
            n = int(phrase.split()[-2])
            assert phrase in (f"within {n} steps", f"for the next {n} steps") and d == s + n
        kinds.add((until, kind))
        assert d > s
    assert kinds == {(False, k) for k in ("nightfall", "dawn", "noon", "step", "within")} | \
        {(True, k) for k in ("nightfall", "dawn", "step", "within")}


# --- perception ------------------------------------------------------------------------------------------
def test_memory_ground_truth_counts_only_perceivable_tiles():
    env = make_env()
    clear_creatures(env)
    b = env.bridge
    x, y = int(player(env).pos[0]), int(player(env).pos[1])
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            world(env)[(x + dx, y + dy)] = "grass"
    inside = [(4, 3), (-4, -3), (0, 1)]
    outside = [(5, 0), (0, 4), (-5, -3), (0, -4)]
    for dx, dy in inside + outside:
        world(env).add(objects.Cow(world(env), (x + dx, y + dy)))
    seen = {tuple(p) for p in perceived_now(b).get("cow", [])}
    assert seen == {(x + dx, y + dy) for dx, dy in inside}
    world(env).daylight = 0.0   # full darkness: the noise vignette hides the edge of the frame
    seen_dark = {tuple(p) for p in perceived_now(b).get("cow", [])}
    assert seen_dark == {(x, y + 1)}
    pos, vis = perceivable_grid(b)
    assert vis.shape == (9, 7) and 5 <= vis.sum() < 63 and vis[4, 3]
    world(env).daylight = 0.49
    assert perceivable_grid(b)[1].sum() > vis.sum()
    world(env).daylight = 0.5
    assert perceivable_grid(b)[1].sum() == 63


def test_visibility_log_is_what_the_report_grader_uses():
    env = make_env()
    clear_creatures(env)
    x, y = int(player(env).pos[0]), int(player(env).pos[1])
    world(env)[(x + 5, y)] = "grass"
    for c in ((x + 6, y), (x + 5, y - 1), (x + 5, y + 1), (x + 4, y)):
        world(env)[c] = "stone"                           # pen it in so it cannot wander into view
    world(env).add(objects.Cow(world(env), (x + 5, y)))   # just outside the 9-wide window
    step(env)
    assert "cow" not in env.layer.state["visibility"]["last_seen"]
    q = env.issue("report", {"question": "last_seen", "kind": "cow"}, deadline=20)
    step(env)
    step(env, "noop", f"answer {q['id']}: never")
    assert q["status"] == "fulfilled"


# --- utterance grammar -----------------------------------------------------------------------------------
def test_utterance_grammar():
    cl = parse("answer q7: 5 east 2 north; ack o3\nhello there; ANSWER q8 three. ; answer q9:north-east far")
    assert [c["kind"] for c in cl] == ["answer", "ack", "say", "answer", "answer"]
    assert cl[0]["payload"] == {"type": "offset", "dx": 5, "dy": -2}
    assert cl[1]["id"] == "o3"
    assert cl[3]["payload"] == {"type": "int", "value": 3}
    assert cl[4]["payload"] == {"type": "bucket", "dir": "north-east", "dist": "far"}
    assert parse_payload("here") == {"type": "offset", "dx": 0, "dy": 0}
    assert parse_payload("2 north 4 west") == {"type": "offset", "dx": -4, "dy": -2}
    assert parse_payload("3 east 1 west")["type"] == "invalid"
    assert parse_payload("Never") == {"type": "none"} and parse_payload("nothing") == {"type": "none"}
    assert parse_payload("sw") == {"type": "bucket", "dir": "south-west", "dist": None}
    assert parse_payload("o12") == {"type": "id", "value": "o12"}
    assert parse_payload("The stone pickaxe!") == {"type": "text", "value": "stone pickaxe"}
    assert parse(None) == [] and parse("") == []
    assert compass(5, 0) == "east" and compass(0, -5) == "north" and compass(-3, 3) == "south-west"
    assert compass(1, -1) == "here" and compass(5, -2) == "east" and compass(5, -3) == "north-east"
    assert distance_bucket(4, 1) == "near" and distance_bucket(-12, 0) == "mid" and distance_bucket(0, 13) == "far"
    assert format_offset(5, -2) == "5 east 2 north" and format_offset(0, 0) == "here"
    for dx in range(-15, 16):
        for dy in range(-15, 16):
            assert parse_payload(format_offset(dx, dy)) == {"type": "offset", "dx": dx, "dy": dy}


# --- scoring ---------------------------------------------------------------------------------------------
def test_official_crafter_score_formula():
    assert crafter_score([10.0] * 22) == pytest.approx(10.0)
    assert crafter_score([100.0] * 11 + [0.0] * 11) == pytest.approx(math.sqrt(101.0) - 1.0)  # 9.04987...
    assert crafter_score([0.0] * 22) == 0.0 and crafter_score([100.0] * 22) == pytest.approx(100.0)
    rates = [90.0, 50.0, 0.0, 1.0] + [0.0] * 18
    hand = math.exp((math.log(91) + math.log(51) + math.log(2)) / 22) - 1.0
    assert crafter_score(rates) == pytest.approx(hand)


def test_aggregate_pools_episodes_and_keeps_components():
    env = W1Env(W1Config(episode_length=150, arrival_rate0=1 / 20))
    sums = []
    for seed in (1, 2):
        obs = env.reset(seed)
        agent = RandomAgent(seed)
        done = False
        while not done:
            a, u = agent.act(obs)
            obs, r, done, info = env.step(a, u)
        sums.append(env.episode_summary())
    ag = aggregate(sums)
    assert ag["episodes"] == 2 and set(ag["index_components"]) == {"crafter_score/100", "value_weighted_on_time",
                                                                   "memory_accuracy", "survival_fraction"}
    rates = ag["crafter"]["success_rates"]
    assert len(rates) == 22 and all(r in (0.0, 50.0, 100.0) for r in rates.values())
    assert ag["crafter"]["score"] == pytest.approx(crafter_score(rates.values()))


def test_baselines_run_and_the_privileged_agent_answers():
    env = W1Env(W1Config(episode_length=400, arrival_rate0=1 / 25))
    obs = env.reset(4)
    agent = ScriptedAgent(4)
    info = env.privileged_now()
    for _ in range(400):
        a, u = agent.act(obs, info)
        obs, r, d, info = env.step(a, u)
        if d:
            break
    s = env.episode_summary()
    assert s["crafter"]["n_unlocked"] >= 5
    assert s["memory"]["asked"] == 0 or s["memory"]["accuracy"] == 1.0
