"""The individual's learned skill faculty: the predictive core and the skill policy, living in the canonical store.

``CoreSkillHost`` binds the ``skills`` slot in place of the routine library.  Everything it is lives in the store:

- ``skills.weights``    the core and policy it acts with (the bytes a consolidation event installed);
- ``skills.core_state`` the core's recurrent state after this tick (its present moment: deter and stoch);
- ``skills.state``      its bookkeeping and its lineage: every consolidation event that made it what it is.

Each tick it reads the frame and its body sense, steps the core one moment (``PredictiveCore.step``, the
deterministic mode, so a life replays exactly), and acts greedily with the policy.  The P-A arm also reads its
own memory (belief features without routine state, and the remembered map around it).  Nothing is kept in Python
between ticks except a decoded copy of the weights, keyed by the stored array, so a restore or a consolidation
is seen at once.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import torch

from continuous_brain_individual import codec, ledger
from continuous_brain_individual import memory as mem
from continuous_brain_individual.body import read_body
from continuous_brain_individual.features import belief_features
from continuous_brain_individual.hosts import _blob_decl, _Host
from continuous_brain_unified.channels import ChannelDecl

from .archive import INVENTORY
from .data import body_vector
from .policy import SkillPolicy
from .record import VITALS, map_crop
from .world_model import PredictiveCore


def pack_weights(core: PredictiveCore, policy: SkillPolicy, chooser=None) -> bytes:
    """The bytes a consolidation installs: the core (without its optimiser), the policy, and (CB2-L, L-C) the
    learned chooser if it has one."""
    buf = io.BytesIO()
    c = core.state_dict_full()
    c.pop("opt", None)
    pack = {"format": "cb2-skill-weights/1", "core": c, "policy": policy.state_dict_full()}
    if chooser is not None:
        pack["chooser"] = chooser.state_dict_full()
    torch.save(pack, buf)
    return buf.getvalue()


def unpack_chooser(blob):
    """The learned chooser in a weights pack, or None (every pack before CB2-L)."""
    saved = torch.load(io.BytesIO(np.asarray(blob, np.uint8).tobytes()), map_location="cpu", weights_only=False)
    if "chooser" not in saved:
        return None
    from .chooser import Chooser
    ch = Chooser.from_state(saved["chooser"])
    ch.eval()
    return ch


def unpack_weights(blob) -> tuple[PredictiveCore, SkillPolicy]:
    saved = torch.load(io.BytesIO(np.asarray(blob, np.uint8).tobytes()), map_location="cpu", weights_only=False)
    core = PredictiveCore.from_state(saved["core"])
    pol = SkillPolicy.from_state(saved["policy"])
    core.eval()
    pol.eval()
    return core, pol


class CoreSkillHost(_Host):
    slot, mechanism = "skills", "cb2-predictive-core-skill"

    def __init__(self):
        super().__init__({"core": "DreamerV3 RSSM (vendored dreamerv3-torch, MIT)", "policy": "imitation, P-A/P-B"})
        self.reads = ("driver.frame", "driver.body", "episodic_memory.state", "goal_genesis.state")
        self.reads_delayed = ("skills.state", "skills.weights", "skills.core_state")
        self.writes = (_blob_decl("skills.state", self.slot, "learned skill bookkeeping and lineage (codec)"),
                       _blob_decl("skills.weights", self.slot, "the installed core and policy (torch bytes)"),
                       _blob_decl("skills.core_state", self.slot, "the core's recurrent state (float32 bytes)"),
                       ChannelDecl("skills.selected_option", self.slot, (1,), "float64",
                                   description="the policy's guess at the routine option it is serving"),
                       ChannelDecl("skills.action", self.slot, (1,), "int64", neutral=0,
                                   description="the Crafter action index chosen this tick"))
        self._cache = (None, None)
        self._chooser = None

    def _models(self, w: np.ndarray):
        key = w.base if w.base is not None else w            # the stored array behind this tick's read view
        if self._cache[0] is not key:
            self._cache = (key, unpack_weights(w))
            self._chooser = unpack_chooser(w)
        return self._cache[1]

    def step(self, ctx):
        st = dict(codec.unblob(ctx.delayed["skills.state"]) or {"lineage": []})
        w = np.asarray(ctx.delayed["skills.weights"])
        body = read_body(ctx.inputs["driver.body"])
        life, step = int(body.get("life", 0)), int(body["step"])
        if w.size == 0:                                        # never consolidated: it cannot act yet
            return {"skills.state": codec.blob(st), "skills.weights": ctx.delayed["skills.weights"],
                    "skills.core_state": ctx.delayed["skills.core_state"],
                    "skills.selected_option": np.zeros(1), "skills.action": np.zeros(1, np.int64)}
        core, pol = self._models(ctx.delayed["skills.weights"])
        first = st.get("life") != life or not st.get("t")
        inv = body["inventory"]
        bv = body_vector([[inv[k] for k in INVENTORY]], [body["sleeping"]], [body["facing"]], [step],
                         [body["player_pos"]])
        prev = None
        if not first:
            raw = np.frombuffer(np.asarray(ctx.delayed["skills.core_state"], np.uint8).tobytes(), np.float32)
            d = core.cfg.deter
            prev = {"deter": torch.from_numpy(raw[:d].copy())[None],
                    "stoch": torch.from_numpy(raw[d:].copy()).reshape(1, core.cfg.stoch, core.cfg.discrete)}
            prev["logit"] = torch.zeros_like(prev["stoch"])
        post = core.step(prev, [0 if first else int(st["prev_action"])],
                         np.asarray(ctx.inputs["driver.frame"], np.uint8)[None], bv, [float(first)])
        feat = core.feat(post)
        hist = ([] if first else list(st.get("hist", []))) + [[int(inv[k]) for k in VITALS]]
        hist = hist[-11:]
        feats = crop = intent = None
        memory = led = None
        selector = getattr(pol.cfg, "head", "action") == "command"
        if pol.cfg.arm == "P-A" or pol.cfg.intent_dim or selector:
            memory = codec.unblob(ctx.inputs["episodic_memory.state"]) or mem.empty()
            led = codec.unblob(ctx.inputs["goal_genesis.state"]) or ledger.empty()
        if pol.cfg.arm == "P-A":
            feats = belief_features(memory, body, step, led, None, hist)[None].astype(np.float16).astype(np.float32)
            crop = map_crop(memory, body["player_pos"]).reshape(1, -1)
        if pol.cfg.intent_dim:
            # r3-T DIAGNOSTIC ONLY: the routine's private intent on this individual's own beliefs, with the
            # routine's plan state kept in skills.state (as the routine host keeps it), fresh each life
            from .intent import IntentLibrary
            lib = IntentLibrary.from_state(None if first else st.get("shadow"), seed=0)
            _, _, vec = lib.decide_with_intent(memory, body, step, ledger.active(led, step))
            st["shadow"] = lib.to_state()
            intent = vec[None]
        with torch.no_grad():
            logits, opt_logits = pol(feat, bv, feats, crop, intent)
        option = int(opt_logits.argmax(-1)[0])
        if selector:
            # r3-H: the learned level chooses the command; the scripted executor organ carries it out on this
            # individual's own beliefs, with its plan state in skills.state.  Every delegated action is logged:
            # the command, the primitive action, the organ's status, and running counts per command.
            from continuous_brain_individual.skills import ACTIONS
            from .organ import COMMANDS, ExecutorOrgan
            option = int(logits.argmax(-1)[0])
            if getattr(pol.cfg, "temperature", 0.0) > 0:        # S2 practice only: a reproducible sample
                p = torch.softmax(logits[0] / pol.cfg.temperature, -1).double().numpy()
                rng = np.random.default_rng([int(pol.cfg.sample_salt), life, step])
                option = int(rng.choice(len(p), p=p / p.sum()))
            chosen = None
            ch = getattr(self, "_chooser", None)
            if ch is not None and ch.cfg.enabled and not getattr(self, "chooser_off", False) \
                    and not getattr(pol.cfg, "temperature", 0.0):
                # CB2-L (L-C): the command its imagination scores best, with the selector's preference as a prior
                option, chosen = ch.choose(core, post, logits[0])
            self.last = {"feat": feat[0].detach().numpy(), "body": np.asarray(bv[0], np.float32), "features": feats[0],
                         "map_crop": crop[0], "logits": logits[0].detach().numpy(), "command": option,
                         "chooser": chosen}
            organ = ExecutorOrgan(0)
            organ.state = None if first else st.get("organ")
            name, tel = organ(COMMANDS[option], memory, body, step, ledger.active(led, step))
            st["organ"] = organ.state
            provenance = "scripted-executor-organ"
            hook = getattr(self, "override", None)        # CB2-H1 Phase 0 tester only (never set in deployment)
            if hook is not None:
                alt = hook(memory, body, step, name)
                if alt is not None and alt != name:
                    tel = {**tel, "overridden": name, "action": alt}
                    name, provenance = alt, "phase0-tester-override"
            counts = {} if first else dict(st.get("delegated", {}))
            counts[COMMANDS[option]] = counts.get(COMMANDS[option], 0) + 1
            choice = {} if chosen is None else {"command_by": "learned-value-choice", "changed": chosen["changed"]}
            st["delegated"], st["delegation"] = counts, {**tel, **choice, "step": step, "provenance": provenance}
            action = ACTIONS[name]
        else:
            action = int(logits.argmax(-1)[0])
        state_bytes = np.concatenate([post["deter"][0].numpy().ravel(), post["stoch"][0].numpy().ravel()]).astype(
            np.float32).tobytes()
        st.update(life=life, t=int(st.get("t", 0)) + 1, prev_action=action, hist=hist)
        return {"skills.state": codec.blob(st), "skills.weights": ctx.delayed["skills.weights"],
                "skills.core_state": np.frombuffer(state_bytes, np.uint8).copy(),
                "skills.selected_option": np.array([option], float),
                "skills.action": np.array([action], dtype=np.int64)}


def weights_sha(blob) -> str:
    return hashlib.sha256(np.asarray(blob, np.uint8).tobytes()).hexdigest()
