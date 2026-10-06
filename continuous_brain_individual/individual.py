"""The persistent individual living in W1 (exploration lane, engineering only).

    ind = Individual.create(IndividualConfig(), vision=VisionOrgan(...), language=LanguageOrgan(...))
    env = W1Env(W1Config()); obs = env.reset(seed)
    life = ind.live(env, obs, info=None, steps=3000)          # or step by step: ind.step(env, obs, info)

Each step the life loop writes what the individual can sense into the store's driver channels -- the pixels
(``driver.frame``), its body sense (``driver.body``: vitals, inventory, position, facing, sleeping, its own
achievement counters, and the clock) and the caretaker's messages (``driver.messages``) -- runs one runtime
tick, and acts in W1 with the action and utterance the individual wrote (``skills.action``,
``metacognition.utterance``).  ``info`` is wrapped so that nothing but the body sense can be read from it.

Arms (``IndividualConfig``) for ablations: ``memory`` "full" | "none"; ``language`` "organ" | "none";
``perception`` "organ" | "oracle" (ground-truth tiles of the current view: isolates perception errors; the only
arm that reads privileged state, and it is labelled so).

Persistence: ``save()`` returns the runtime state (the individual) and the world snapshot; ``restore()``
continues bit-exactly.  Interventions on any ``<slot>.state`` channel (ERASE, REPLACE, ...) are registered before
the first tick through ``IndividualConfig.interventions``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from continuous_brain_protocols import InterventionKind, InterventionSpec
from continuous_brain_unified.channels import ChannelDecl
from continuous_brain_unified.runtime import BindingPolicy, RuntimeConfig, UnifiedRuntime
from continuous_brain_organs.vision import labels_from_privileged

from . import codec
from .body import GuardedInfo, body_message, body_sense
from .hosts import (AutobiographyHost, LedgerHost, MemoryHost, PerceptionHost, SelfModelHost, SkillHost,
                    SpeechHost)

DRIVER = (ChannelDecl("driver.frame", "driver", (64, 64, 3), "uint8", description="W1 pixels"),
          ChannelDecl("driver.body", "driver", None, "uint8", description="body sense + clock (JSON)"),
          ChannelDecl("driver.messages", "driver", None, "uint8", description="caretaker messages this step (JSON)"),
          ChannelDecl("driver.oracle_tiles", "driver", None, "uint8",
                      description="ground-truth tiles of the view: ORACLE ablation only, else empty"))


@dataclass
class IndividualConfig:
    memory: str = "full"
    language: str = "organ"
    perception: str = "organ"
    skills: str = "routine"           # "routine": the borrowed routine library; "learned": foresight + learned choice;
                                      # "cb2": the CB2-S1 predictive core and skill policy (installed by consolidation)
    learn: bool = True                # learned skills only: False freezes what it has learned (evaluation)
    lifelong: bool = False            # one individual across lives: it knows its life number and its dying moment
    seed: int = 0
    interventions: list = field(default_factory=list)      # [{"kind", "target", "start_tick", "end_tick", ...}]

    def arm(self) -> str:
        parts = []
        if self.perception != "organ":
            parts.append(f"perception={self.perception}")
        if self.memory != "full":
            parts.append(f"memory={self.memory}")
        if self.language != "organ":
            parts.append(f"language={self.language}")
        if self.skills != "routine":
            parts.append(f"skills={self.skills}" + ("" if self.learn else ",frozen"))
        return "individual" + ("[" + ",".join(parts) + "]" if parts else "")


class Individual:
    def __init__(self, config: IndividualConfig, vision, language, log_dir=None):
        self.config, self.vision, self.language = config, vision, language
        rt = UnifiedRuntime(RuntimeConfig(seed=config.seed, binding_policy=BindingPolicy.TEST_FIXTURES,
                                          driver=DRIVER), log_dir=log_dir)
        hosts = [PerceptionHost(vision, oracle=config.perception == "oracle"), MemoryHost(config.memory),
                 LedgerHost(language, config.language), SelfModelHost(), SpeechHost(language, config.language),
                 AutobiographyHost(language)]
        if config.skills == "learned":
            from .learned import LearnedSkillHost, WorldModelHost
            hosts += [WorldModelHost(config.seed, config.learn), LearnedSkillHost(config.seed, config.learn)]
        elif config.skills == "routine":
            hosts.append(SkillHost(config.seed))
        elif config.skills == "cb2":                      # the CB2-S1 learned skill: predictive core + policy
            from continuous_brain_cb2.skill_host import CoreSkillHost
            hosts.append(CoreSkillHost())
        else:
            raise ValueError(f"unknown skills {config.skills!r}")
        rt.bind_many(hosts)
        for iv in config.interventions:
            rt.interventions.register(InterventionSpec(InterventionKind(iv["kind"]), iv["target"],
                                                       int(iv["start_tick"]), iv.get("end_tick"),
                                                       iv.get("payload_ref"), note=iv.get("note", "")))
            if iv.get("payload") is not None:
                rt.interventions.add_payload(iv["payload_ref"], iv["payload"])
        self.rt = rt
        self.life = 0                 # which life (lifelong): the world's news, given to it by the driver

    # ------------------------------------------------------------------------------------------------- life
    def sense(self, obs: dict, info: dict | None, dying: bool = False) -> dict:
        if info is None:                       # before the first world step: the reset observation
            raise ValueError("the individual needs W1's info (for its body sense) at every step")
        body = GuardedInfo(info).body()
        oracle = b""
        if self.config.perception == "oracle":
            mat, objs = labels_from_privileged(info["privileged"])
            oracle = codec.encode({"material": mat, "object": objs,
                                   "material_conf": np.ones(mat.shape, np.float32),
                                   "object_conf": np.ones(objs.shape, np.float32)})
        return {"driver.frame": np.asarray(obs["image"], dtype=np.uint8),
                "driver.body": body_message(body, int(obs["step"]), self.life if self.config.lifelong else None,
                                            dying),
                "driver.messages": json.dumps(list(obs["messages"])).encode(),
                "driver.oracle_tiles": oracle}

    def decide(self, obs: dict, info: dict) -> tuple[int, str | None]:
        action, text, _ = self.tick(obs, info)
        return action, text

    def tick(self, obs: dict, info: dict, dying: bool = False):
        """One runtime tick: (action, utterance or None, the runtime's TickResult with each channel's digest)."""
        result = self.rt.tick_once(driver=self.sense(obs, info, dying))
        action = int(np.asarray(self.rt.store.read("skills.action"))[0])
        text = np.asarray(self.rt.store.read("metacognition.utterance"), dtype=np.uint8).tobytes().decode()
        return action, (text or None), result

    def die(self, obs: dict, info: dict):
        """Its last moment (lifelong): one more tick in which it knows it is dying; nothing it does reaches the
        world.  Its faculties close the life: the last decision's return ends in death, it consolidates what it
        learned, its autobiography records the life."""
        if not self.config.lifelong:
            raise ValueError("only a lifelong individual has a last moment")
        return self.tick(obs, info, dying=True)[2]

    def next_life(self) -> None:
        if not self.config.lifelong:
            raise ValueError("only a lifelong individual continues into another life")
        self.life += 1

    def live(self, env, obs: dict, info: dict, steps: int, on_step=None) -> dict:
        done = False
        n = 0
        while n < steps and not done:
            action, utter = self.decide(obs, info)
            obs, reward, done, info = env.step(action, utter)
            n += 1
            if on_step is not None:
                on_step(n, action, utter, obs, reward, info)
        return {"steps": n, "done": done, "obs": obs, "info": info}

    # ------------------------------------------------------------------------------------------ persistence
    def save(self) -> dict:
        return {"runtime": self.rt.state_dict(), "life": self.life}

    def restore(self, saved: dict) -> None:
        self.rt.load_state_dict(saved["runtime"])
        self.life = int(saved.get("life", 0))

    def state_hash(self) -> str:
        return self.rt.state_hash()

    def read_state(self, slot: str):
        return codec.unblob(self.rt.store.read(f"{slot}.state"))


def first_info(env) -> dict:
    """W1's info for the reset observation (its body sense), via the env's own privileged snapshot."""
    return env.privileged_now()
