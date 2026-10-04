"""The individual's learned faculties: foresight (world_model) and choice (skills), trained on its own lives.

Both learn only from the individual's own experience: what it believed (``features.belief_features``), what it
chose, and what it then felt through its body sense and heard from the caretaker.  No teacher, no privileged
state.  Everything they learn lives in the canonical store:

    world_model.state    bookkeeping (codec): pending foresight samples, its vitals history, counters, losses
    world_model.weights  float32: the foresight network, Adam's moments
    world_model.memory   float16: resolved foresight samples (features, what then happened), a ring
    world_model.features float64: this tick's belief features (read by skills)
    world_model.prediction float64: predicted risks for the next HORIZON steps (canonical channel)

    skills.state    bookkeeping (codec): the routine library's state, the current option, returns, counters
    skills.weights  float32: the option-value network, its target copy, Adam's moments
    skills.memory   float16: its replay memory of decisions (state, feasible options, option, return,
                    duration, terminal), one row per decision; a row's next state is the following row
    skills.selected_option, skills.action, skills.values (the option values it chose from)

Foresight.  Every SAMPLE_EVERY steps it notes its belief features; HORIZON steps later (or when it dies) it
learns what followed: did it die, run out of water or food, lose 3 or more health, or run out of energy.  A
small network predicts those risks from the features (logistic loss), so what it believes now carries what it
has learned about where that leads.

Choice.  An option is a temporally extended course of action from the skill library (skills.OPTIONS): its
routine, drinking, eating, resting in a shelter, retreating from threats, exploring, its requests, or building.
At each decision point (the option ended, could not go on, or something salient happened: it was hurt, water
or food fell to 3, night fell, the caretaker asked something new) it chooses among the feasible options by a
Boltzmann distribution over learned option values plus a prior that favours its routine (so it starts where the
routine is and learns when to depart from it).  The values are learned by semi-Markov double Q-learning over its
replay memory: the return of an option is the discounted sum of what it felt while the option ran (health
gained or lost, new things done, the caretaker's thanks or complaints) plus, at a decision, the discounted value
of the next state; dying ends the return with DEATH.

When it learns: a few updates after each decision, more while it sleeps, and a longer consolidation in the
moment of dying.  With ``learn=False`` (evaluation) it chooses with its lowest temperature and does not learn.
"""
from __future__ import annotations

import numpy as np

from continuous_brain_unified.channels import ChannelDecl
from continuous_brain_world import daylight as dl

from . import codec, ledger, memory as mem
from .body import read_body
from .features import DIM, belief_features
from .hosts import _blob_decl, _Host, _json, state_channel
from .learning import Adam, F, Layout, bce_grad, huber_grad, mlp_backward, mlp_forward, mlp_init, mlp_shapes, sigmoid
from .skills import ACTIONS, OPTIONS, SkillLibrary

# ------------------------------------------------------------------------------------------------- foresight
HORIZON = 100
SAMPLE_EVERY = 5
RISKS = ("die", "run out of water", "run out of food", "lose 3+ health", "run out of energy")
K = len(RISKS)
WM_SIZES = [DIM, 48, 48, K]
WM_CAP = 2048
WM_ROW = DIM + K
WM_NET = mlp_shapes(WM_SIZES)
WM_W = Layout([(f"p_{n}", s) for n, s in WM_NET] + [(f"m_{n}", s) for n, s in WM_NET]
              + [(f"v_{n}", s) for n, s in WM_NET])
WM_LR, WM_BATCH, WM_CONSOLIDATE, WM_MIN = 1e-3, 64, 300, 128

# ---------------------------------------------------------------------------------------------------- choice
NOPT = len(OPTIONS)
Q_IN = DIM + K
Q_SIZES = [Q_IN, 96, 96, NOPT]
Q_NET = mlp_shapes(Q_SIZES)
Q_W = Layout([(f"p_{n}", s) for n, s in Q_NET] + [(f"t_{n}", s) for n, s in Q_NET]
             + [(f"m_{n}", s) for n, s in Q_NET] + [(f"v_{n}", s) for n, s in Q_NET])
Q_CAP = 2048
# replay row: state, feasible mask, option, return, duration, terminal, complete
R_STATE, R_MASK = slice(0, Q_IN), slice(Q_IN, Q_IN + NOPT)
R_OPT, R_RET, R_DUR, R_DONE, R_OK = Q_IN + NOPT, Q_IN + NOPT + 1, Q_IN + NOPT + 2, Q_IN + NOPT + 3, Q_IN + NOPT + 4
Q_ROW = Q_IN + NOPT + 5
GAMMA, DEATH = 0.995, -5.0
Q_LR, Q_BATCH, Q_PER_DECISION, Q_PER_SLEEP_TICK, Q_CONSOLIDATE, Q_MIN = 5e-4, 64, 4, 1, 600, 64
TARGET_EVERY = 200
TAU0, TAU_MIN, TAU_HALF = 0.5, 0.05, 1500.0            # temperature halves every TAU_HALF decisions
PRIOR = np.array([0.7] + [0.3 / (NOPT - 1)] * (NOPT - 1))
FEEDBACK_REWARD = {"fulfilled": 1.0, "correct": 1.0, "late": -1.0, "violated": -1.0, "wrong": -1.0,
                   "no_answer": -1.0}


def _views(layout: Layout, flat: np.ndarray, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in layout.views(flat).items() if k.startswith(prefix)}


def _init_weights(layout: Layout, sizes: list[int], rng, with_target: bool) -> np.ndarray:
    flat = layout.zeros()
    p = mlp_init(rng, sizes)
    for k, v in p.items():
        layout.view(flat, f"p_{k}")[...] = v
        if with_target:
            layout.view(flat, f"t_{k}")[...] = v
    return flat


def _sample(rng, n_valid: np.ndarray, size: int) -> np.ndarray:
    return n_valid[rng.integers(0, len(n_valid), size=size)]


# =============================================================================================== foresight host
class WorldModelHost(_Host):
    slot, mechanism = "world_model", "learned-foresight"

    def __init__(self, seed: int = 0, learn: bool = True):
        super().__init__({"risks": list(RISKS), "horizon": HORIZON, "sizes": WM_SIZES, "learn": learn})
        self.seed, self.learn = int(seed), bool(learn)
        self.reads = ("episodic_memory.state", "goal_genesis.state", "driver.body")
        self.reads_delayed = (state_channel(self.slot), "world_model.weights", "world_model.memory", "skills.state")
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "foresight bookkeeping (codec)"),
                       ChannelDecl("world_model.weights", self.slot, (WM_W.size,), "float32",
                                   description="foresight network and Adam moments"),
                       ChannelDecl("world_model.memory", self.slot, (WM_CAP * WM_ROW,), "float16",
                                   description="resolved foresight samples: features, what followed"),
                       ChannelDecl("world_model.features", self.slot, (DIM,), "float64",
                                   description="this tick's belief features"),
                       ChannelDecl("world_model.prediction", self.slot, (K,), "float64",
                                   description=f"predicted chance, within {HORIZON} steps, to: " + ", ".join(RISKS)))

    def step(self, ctx):
        st = codec.unblob(ctx.delayed[state_channel(self.slot)])
        weights = np.asarray(ctx.delayed["world_model.weights"])
        buf = np.asarray(ctx.delayed["world_model.memory"])
        body = read_body(ctx.inputs["driver.body"])
        step, life, dying = int(body["step"]), int(body.get("life", 0)), bool(body.get("dying", False))
        inv = body["inventory"]
        if st is None or not weights.any():                     # first tick, or its learning was erased
            rng = np.random.default_rng(self.seed * 7919 + 101)
            weights = _init_weights(WM_W, WM_SIZES, rng, with_target=False)
            buf = np.zeros(WM_CAP * WM_ROW, np.float16)
            st = {"life": life, "t": 0, "head": 0, "count": 0, "pending": [], "hist": [], "samples": 0,
                  "loss": None, "brier": [None] * K, "rng": rng.bit_generator.state, "updates": 0}
        st = dict(st)
        if st["life"] != life:                                  # a new world: what it was waiting to learn is gone
            st["pending"], st["hist"], st["life"] = [], [], life
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng"]
        memory = codec.unblob(ctx.inputs["episodic_memory.state"]) or mem.empty()
        led = codec.unblob(ctx.inputs["goal_genesis.state"]) or ledger.empty()
        sk = codec.unblob(ctx.delayed["skills.state"]) or {}
        vit = [int(inv[k]) for k in ("health", "food", "drink", "energy")]
        hist = (list(st["hist"]) + [vit])[-11:]
        phi = belief_features(memory, body, step, led, sk.get("lib"), hist)
        # what followed each pending sample
        resolved, pending = [], []
        for p in st["pending"]:
            flags = list(p["flags"])
            flags[1] = max(flags[1], float(inv["drink"] == 0))
            flags[2] = max(flags[2], float(inv["food"] == 0))
            flags[3] = max(flags[3], float(inv["health"] <= p["h0"] - 3))
            flags[4] = max(flags[4], float(inv["energy"] == 0))
            if dying:
                flags[0] = 1.0
            if dying or step - p["step"] >= HORIZON:
                resolved.append((np.asarray(p["row"], np.float32), np.asarray(flags, np.float32)))
            else:
                pending.append({**p, "flags": flags})
        if not dying and step % SAMPLE_EVERY == 0:
            pending.append({"step": step, "row": phi.astype(np.float32), "h0": int(inv["health"]), "flags": [0.0] * K})
        if resolved:
            buf = np.array(buf, copy=True)
            rows = buf.reshape(WM_CAP, WM_ROW)
            P = _views(WM_W, weights, "p_")
            brier = list(st["brier"])
            for x, y in resolved:
                pred = sigmoid(mlp_forward(P, x[None], 3)[0][0])     # how well it had foreseen this
                for i in range(K):
                    e = float((pred[i] - y[i]) ** 2)
                    brier[i] = e if brier[i] is None else 0.99 * brier[i] + 0.01 * e
                rows[st["head"]] = np.concatenate([x, y]).astype(np.float16)
                st["head"], st["count"] = (st["head"] + 1) % WM_CAP, min(WM_CAP, st["count"] + 1)
                st["samples"] += 1
            st["brier"] = brier
        n_updates = 0
        if self.learn and st["count"] >= WM_MIN:
            n_updates = (1 if resolved else 0) + (2 if body["sleeping"] else 0) + (WM_CONSOLIDATE if dying else 0)
        if n_updates:
            weights = np.array(weights, copy=True)
            rows = buf.reshape(WM_CAP, WM_ROW)
            P, M, V = (_views(WM_W, weights, p) for p in ("p_", "m_", "v_"))
            opt = Adam(lr=WM_LR)
            loss = st["loss"]
            for _ in range(n_updates):
                idx = rng.integers(0, st["count"], size=WM_BATCH)
                batch = rows[idx].astype(np.float32)
                x, y = batch[:, :DIM], batch[:, DIM:]
                out, cache = mlp_forward(P, x, 3)
                l, d = bce_grad(out, y)
                st["t"] += 1
                opt.step(P, mlp_backward(P, cache, d, 3), M, V, st["t"])
                loss = l if loss is None else 0.99 * loss + 0.01 * l
            st["loss"], st["updates"] = loss, st["updates"] + n_updates
        P = _views(WM_W, weights, "p_")
        prediction = sigmoid(mlp_forward(P, phi[None], 3)[0][0]).astype(np.float64)
        st["pending"], st["hist"], st["rng"] = pending, hist, rng.bit_generator.state
        return {state_channel(self.slot): codec.blob(st), "world_model.weights": weights, "world_model.memory": buf,
                "world_model.features": phi.astype(np.float64), "world_model.prediction": prediction}


# ================================================================================================= choice host
class LearnedSkillHost(_Host):
    slot, mechanism = "skills", "learned-option-selector"

    def __init__(self, seed: int = 0, learn: bool = True):
        super().__init__({"library": "W1 reference routines over the individual's beliefs, as options",
                          "options": list(OPTIONS), "selector": "semi-Markov double Q-learning", "learn": learn,
                          "sizes": Q_SIZES, "gamma": GAMMA, "death": DEATH})
        self.seed, self.learn = int(seed), bool(learn)
        self.reads = ("episodic_memory.state", "goal_genesis.state", "goal_genesis.events", "driver.body",
                      "world_model.features", "world_model.prediction")
        self.reads_delayed = (state_channel(self.slot), "skills.weights", "skills.memory")
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "routine state and the selector's bookkeeping"),
                       ChannelDecl("skills.weights", self.slot, (Q_W.size,), "float32",
                                   description="option-value network, its target copy, Adam moments"),
                       ChannelDecl("skills.memory", self.slot, (Q_CAP * Q_ROW,), "float16",
                                   description="replay memory of its decisions"),
                       ChannelDecl("skills.selected_option", self.slot, (1,), "float64",
                                   description="index of the option it is following (skills.OPTIONS)"),
                       ChannelDecl("skills.action", self.slot, (1,), "int64", neutral=0,
                                   description="the Crafter action index chosen this tick"),
                       ChannelDecl("skills.values", self.slot, (NOPT,), "float64",
                                   description="its learned value of each option now"))

    # ------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def _feeling(prev: dict | None, body: dict, events: list[dict]) -> float:
        """What it felt since the last tick: health gained or lost, new things done, the caretaker's verdicts."""
        if prev is None:
            return 0.0
        inv = body["inventory"]
        done = sum(1 for v in body["achievements"].values() if v)
        r = 0.1 * (inv["health"] - prev["health"]) + 1.0 * max(0, done - prev["done"])
        for e in events:
            if e["type"] == "feedback":
                r += FEEDBACK_REWARD.get(e["outcome"], 0.0)
        return float(r)

    def _learn(self, ctl: dict, weights: np.ndarray, rows: np.ndarray, rng, n: int) -> np.ndarray:
        head, count = ctl["head"], ctl["count"]
        latest = (head - 1) % Q_CAP
        ok = rows[:count, R_OK].astype(bool) if count < Q_CAP else rows[:, R_OK].astype(bool)
        idx = np.nonzero(ok)[0]
        idx = idx[(rows[idx, R_DONE] > 0.5) | (idx != latest)]       # its next state must exist
        if len(idx) < Q_MIN:
            return weights
        weights = np.array(weights, copy=True)
        P, T, M, V = (_views(Q_W, weights, p) for p in ("p_", "t_", "m_", "v_"))
        opt = Adam(lr=Q_LR)
        loss = ctl["loss"]
        for _ in range(n):
            b = _sample(rng, idx, Q_BATCH)
            batch = rows[b].astype(np.float32)
            nxt = rows[(b + 1) % Q_CAP].astype(np.float32)
            s, o = batch[:, R_STATE], batch[:, R_OPT].astype(np.int64)
            R, k, done = batch[:, R_RET], batch[:, R_DUR], batch[:, R_DONE]
            s2, mask2 = nxt[:, R_STATE], nxt[:, R_MASK] > 0.5
            q2_online = mlp_forward(P, s2, 3)[0]
            q2_target = mlp_forward(T, s2, 3)[0]
            a2 = np.argmax(np.where(mask2, q2_online, -1e9), axis=1)
            y = R + (1.0 - done) * (GAMMA ** k) * q2_target[np.arange(len(b)), a2]
            q, cache = mlp_forward(P, s, 3)
            err = q[np.arange(len(b)), o] - y
            l, g = huber_grad(err)
            d = np.zeros_like(q)
            d[np.arange(len(b)), o] = g
            ctl["t"] += 1
            opt.step(P, mlp_backward(P, cache, d, 3), M, V, ctl["t"])
            loss = l if loss is None else 0.99 * loss + 0.01 * l
            if ctl["t"] % TARGET_EVERY == 0:
                for key in P:
                    T[key][...] = P[key]
        ctl["loss"], ctl["updates"] = loss, ctl["updates"] + n
        return weights

    def temperature(self, ctl: dict) -> float:
        if not self.learn:
            return TAU_MIN
        return max(TAU_MIN, TAU0 * 0.5 ** (ctl["decisions"] / TAU_HALF))

    # ---------------------------------------------------------------------------------------------- step
    def step(self, ctx):
        st = codec.unblob(ctx.delayed[state_channel(self.slot)])
        weights = np.asarray(ctx.delayed["skills.weights"])
        buf = np.asarray(ctx.delayed["skills.memory"])
        body = read_body(ctx.inputs["driver.body"])
        step, life, dying = int(body["step"]), int(body.get("life", 0)), bool(body.get("dying", False))
        if st is None or not weights.any():                     # first tick, or its learning was erased
            rng = np.random.default_rng(self.seed * 7919 + 202)
            weights = _init_weights(Q_W, Q_SIZES, rng, with_target=True)
            buf = np.zeros(Q_CAP * Q_ROW, np.float16)
            st = {"lib": None, "ctl": {"life": life, "t": 0, "updates": 0, "decisions": 0, "head": 0, "count": 0,
                                       "opt": None, "row": None, "steps_in": 0, "R": 0.0, "disc": 1.0,
                                       "prev": None, "rng": rng.bit_generator.state, "loss": None, "lives": 0,
                                       "life_return": 0.0, "life_options": [0] * NOPT, "returns": []}}
        ctl = dict(st["ctl"])
        lib_state = st["lib"]
        if ctl["life"] != life:                                 # a new world: its plans there are gone
            lib_state, ctl["life"] = None, life
            ctl.update(opt=None, row=None, steps_in=0, R=0.0, disc=1.0, prev=None, life_return=0.0,
                       life_options=[0] * NOPT)
        rng = np.random.default_rng()
        rng.bit_generator.state = ctl["rng"]
        lib = SkillLibrary.from_state(lib_state, seed=self.seed + life)
        memory = codec.unblob(ctx.inputs["episodic_memory.state"]) or mem.empty()
        led = codec.unblob(ctx.inputs["goal_genesis.state"]) or ledger.empty()
        events = _json(ctx.inputs["goal_genesis.events"]) or []
        s = np.concatenate([np.asarray(ctx.inputs["world_model.features"]),
                            np.asarray(ctx.inputs["world_model.prediction"])]).astype(np.float32)
        P = _views(Q_W, weights, "p_")
        values = mlp_forward(P, s[None], 3)[0][0].astype(np.float64)
        rows = buf.reshape(Q_CAP, Q_ROW)
        writable = False

        def rows_w():
            nonlocal buf, rows, writable
            if not writable:
                buf = np.array(buf, copy=True)
                rows, writable = buf.reshape(Q_CAP, Q_ROW), True
            return rows

        # what it felt since its last action belongs to the option that chose that action
        r = self._feeling(ctl["prev"], body, events) + (DEATH if dying else 0.0)
        if ctl["opt"] is not None:
            ctl["R"] += ctl["disc"] * r
            ctl["disc"] *= GAMMA
            ctl["steps_in"] += 1
        ctl["life_return"] += r
        inv = body["inventory"]
        prev = ctl["prev"]
        ctl["prev"] = {"health": int(inv["health"]), "drink": int(inv["drink"]), "food": int(inv["food"]),
                       "done": sum(1 for v in body["achievements"].values() if v), "dark": bool(dl.is_dark(step))}

        def close(done: bool):
            if ctl["opt"] is None:
                return
            if self.learn and ctl["row"] is not None:
                rw = rows_w()
                rw[ctl["row"], R_RET], rw[ctl["row"], R_DUR] = ctl["R"], max(1, ctl["steps_in"])
                rw[ctl["row"], R_DONE], rw[ctl["row"], R_OK] = float(done), 1.0
            ctl.update(opt=None, row=None, steps_in=0, R=0.0, disc=1.0)

        n_updates = 0
        if dying:
            close(done=True)
            ctl["lives"] += 1
            ctl["returns"] = (list(ctl["returns"]) + [round(ctl["life_return"], 3)])[-200:]
            n_updates = Q_CONSOLIDATE
            name, option = "noop", 0
        else:
            commitments = ledger.active(led, step)
            v = lib.belief_view(memory, body, step, commitments)
            name = None
            option = ctl["opt"] if ctl["opt"] is not None else 0
            if v.sleeping:
                name = "noop"
                n_updates = Q_PER_SLEEP_TICK
            else:
                lib._prepare(v)
                salient = prev is not None and (
                    inv["health"] < prev["health"] or (inv["drink"] <= 3 < prev["drink"])
                    or (inv["food"] <= 3 < prev["food"]) or (dl.is_dark(step) and not prev["dark"])
                    or any(e["type"] in ("new_commitment", "question", "amendment", "cancellation") for e in events))
                if ctl["opt"] is not None and not salient and not lib.option_done(ctl["opt"], v, ctl["steps_in"]):
                    name = lib.routine(v, memory, step)[0] if ctl["opt"] == 0 else lib.option_act(ctl["opt"], v, memory, step)
                if name is None:                                # a decision
                    close(done=False)
                    mask = lib.option_feasible(v, memory)
                    tau = self.temperature(ctl)
                    logits = np.where(mask, values / tau + np.log(PRIOR), -np.inf)
                    p = np.exp(logits - logits.max())
                    p /= p.sum()
                    o = int(rng.choice(NOPT, p=p))
                    name = lib.routine(v, memory, step)[0] if o == 0 else lib.option_act(o, v, memory, step)
                    if name is None:                            # it cannot start: its routine acts instead
                        o = 0
                        name = lib.routine(v, memory, step)[0]
                    ctl.update(opt=o, steps_in=0, R=0.0, disc=1.0)
                    ctl["decisions"] += 1
                    ctl["life_options"] = [c + (i == o) for i, c in enumerate(ctl["life_options"])]
                    if self.learn:
                        rw = rows_w()
                        row = np.zeros(Q_ROW, np.float32)
                        row[R_STATE], row[R_MASK], row[R_OPT] = s, mask.astype(np.float32), o
                        rw[ctl["head"]] = row.astype(np.float16)
                        ctl["row"] = ctl["head"]
                        ctl["head"], ctl["count"] = (ctl["head"] + 1) % Q_CAP, min(Q_CAP, ctl["count"] + 1)
                        n_updates = Q_PER_DECISION
                option = ctl["opt"]
        if self.learn and n_updates:
            weights = self._learn(ctl, weights, rows, rng, n_updates)
        ctl["rng"] = rng.bit_generator.state
        new = {"lib": lib.to_state(), "ctl": ctl}
        return {state_channel(self.slot): codec.blob(new), "skills.weights": weights, "skills.memory": buf,
                "skills.selected_option": np.array([float(option)]),
                "skills.action": np.array([ACTIONS[name]], dtype=np.int64), "skills.values": values}
