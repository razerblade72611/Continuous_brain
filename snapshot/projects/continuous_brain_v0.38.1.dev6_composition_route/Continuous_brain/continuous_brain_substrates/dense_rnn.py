"""DENSE-RNN comparator (Amendment A1.1): a GRU matched to the SNN's budget.

Report-only. It is not part of any PASS gate until David approves A1.

Specification (A1.1 + Joint Contract):
  * trainable parameter count within +-10 % of the SNN's active synapse count;
    the hidden size is solved from that target, never hand-picked
  * the same input channels and the same shared encoder as the SNN, fed as
    continuous values
  * self-supervised objective: predict the encoded sensory channels
    `prediction_horizon_frames` ahead (1 frame = 10 ms at the default dt) from
    the causal history, which includes the current efference copy and never
    any future efference
  * truncated BPTT over `tbptt_frames`, Adam, gradient clipping; all
    hyperparameters live in a frozen config whose hash is recorded
  * single exposure: every target frame is trained on exactly once, enforced
    by a ledger. Re-use needs a replay budget declared in the config (and the
    same budget must be declared for the SNN track)
  * never trained on probe targets; probes read the hidden state only

Stepping is inference-only (no graph). `train_step` replays the current
window from the stored window-start hidden state with a graph, which is the
standard TBPTT approximation (the carried hidden state was computed with the
pre-update parameters).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from continuous_brain_protocols import (BudgetReport, Capability, CapabilityUnavailable, FeatureSpec,
                                        LesionRecord, LesionSpec, PlasticityReport, StreamSpec, Substrate,
                                        SubstrateFrozenError, SubstrateMetadata, TrainReport, read_only,
                                        sha256_json, state_digest)

from .encoders import EncoderConfig, SensoryEncoder, motor_stream_spec, sensory_stream_spec
from .memory import peak_rss_bytes

DENSE_VERSION = "dense-rnn-gru/1"


class BudgetMatchError(ValueError):
    pass


class SingleExposureViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class DenseRNNConfig:
    seed: int = 0
    hidden_size: int | None = None               # None: solve from target_learned_scalars
    target_learned_scalars: int | None = None    # total-match target (A1.1)
    target_recurrent_scalars: int | None = None  # recurrent-core target (A1.1b)
    budget_mode: str = "total"                   # total | recurrent_core
    budget_tolerance: float = 0.10
    min_hidden: int = 4
    learning_rate: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    tbptt_frames: int = 20
    prediction_horizon_frames: int = 1
    replay_budget_targets: int = 0               # 0 = strict single exposure
    frame_dt_s: float = 0.01
    device: str = "cpu"

    def config_hash(self) -> str:
        d = asdict(self)
        d.pop("device")                          # the device is recorded in the budget, not the science
        return sha256_json(d)


def gru_parameter_count(hidden: int, n_in: int, n_pred: int, n_motor: int) -> int:
    gru = 3 * hidden * n_in + 3 * hidden * hidden + 6 * hidden
    return gru + hidden * n_pred + n_pred + hidden * n_motor + n_motor




def gru_recurrent_weight_count(hidden: int) -> int:
    """Learned recurrent matrix scalars only (GRUCell.weight_hh = [3H, H]).

    A1.1b matches these weights to SNN cortex->cortex active synapses. Recurrent
    biases are reported but deliberately excluded: an SNN synapse maps to a
    learned weight, not a bias.
    """
    return 3 * hidden * hidden


def solve_hidden_size_recurrent(target: int, tolerance: float = 0.10, min_hidden: int = 4) -> tuple[int, int]:
    if target <= 0:
        raise BudgetMatchError("recurrent target must be positive")
    root = (target / 3.0) ** 0.5
    candidates = {max(min_hidden, int(math.floor(root))), max(min_hidden, int(math.ceil(root)))}
    best = min(candidates, key=lambda h: abs(gru_recurrent_weight_count(h) - target))
    count = gru_recurrent_weight_count(best)
    if abs(count - target) > tolerance * target:
        raise BudgetMatchError(f"closest recurrent GRU core has {count} weights, outside +-{tolerance:.0%} of {target}")
    return best, count

def solve_hidden_size(n_in: int, n_pred: int, n_motor: int, target: int, tolerance: float = 0.10,
                      min_hidden: int = 4) -> tuple[int, int]:
    """Hidden size whose total parameter count is closest to `target`; raises if outside tolerance."""
    a, b, c = 3.0, 3.0 * n_in + 6.0 + n_pred + n_motor, float(n_pred + n_motor - target)
    disc = b * b - 4 * a * c
    if disc < 0:
        raise BudgetMatchError("no real hidden size reaches the target")
    root = (-b + math.sqrt(disc)) / (2 * a)
    candidates = {max(min_hidden, int(math.floor(root))), max(min_hidden, int(math.ceil(root)))}
    best = min(candidates, key=lambda h: abs(gru_parameter_count(h, n_in, n_pred, n_motor) - target))
    count = gru_parameter_count(best, n_in, n_pred, n_motor)
    if abs(count - target) > tolerance * target:
        raise BudgetMatchError(f"closest GRU has {count} parameters, outside +-{tolerance:.0%} of {target} "
                               f"(input/output layers alone need {gru_parameter_count(min_hidden, n_in, n_pred, n_motor)})")
    return best, count


class IntervalLedger:
    """Merged, sorted half-open intervals of *input* frame ids whose prediction targets were trained on."""

    def __init__(self, intervals: list[tuple[int, int]] | None = None):
        self.intervals: list[tuple[int, int]] = [tuple(x) for x in (intervals or [])]

    def overlap(self, start: int, stop: int) -> int:
        return sum(max(0, min(stop, b) - max(start, a)) for a, b in self.intervals)

    def add(self, start: int, stop: int) -> None:
        merged: list[tuple[int, int]] = []
        for a, b in sorted(self.intervals + [(start, stop)]):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.intervals = merged

    def total(self) -> int:
        return sum(b - a for a, b in self.intervals)


class _GRUModel(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_pred: int, n_motor: int):
        super().__init__()
        self.cell = nn.GRUCell(n_in, hidden)
        self.pred = nn.Linear(hidden, n_pred)
        self.motor = nn.Linear(hidden, n_motor)


@dataclass
class TrainBatch:
    first_input: int                 # frame id of the first input in the window
    inputs: torch.Tensor             # [T, n_in] frames first_input .. first_input+T-1
    targets: torch.Tensor            # [T, n_pred] sensory frames first_input+h .. first_input+T-1+h
    h0: torch.Tensor                 # [1, H] hidden state before first_input (detached)

    @property
    def target_ids(self) -> tuple[int, int]:
        return self.first_input, self.first_input + int(self.inputs.shape[0])


class DenseStateFeatures:
    def __init__(self, sub: "DenseRNNSubstrate"):
        self.sub = sub

    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec("dense_hidden", self.sub.hidden_size, "GRU hidden state at the current frame")

    def read_features(self) -> np.ndarray:
        return self.sub.h[0].detach().cpu().numpy().astype(np.float64)


class DenseRNNSubstrate(Substrate):
    def __init__(self, encoder_cfg: EncoderConfig, cfg: DenseRNNConfig):
        self.cfg = cfg
        self.encoder_cfg = encoder_cfg
        self.encoder = SensoryEncoder(encoder_cfg)
        self.device = torch.device(cfg.device)
        self.n_in = self.encoder.input_size()
        self.n_pred = self.encoder.sensory_size()
        self.n_motor = encoder_cfg.n_joints * encoder_cfg.motor_channels
        if cfg.budget_mode not in ("total", "recurrent_core"):
            raise ValueError("budget_mode must be 'total' or 'recurrent_core'")
        target = cfg.target_learned_scalars if cfg.budget_mode == "total" else cfg.target_recurrent_scalars
        if cfg.hidden_size is not None:
            self.hidden_size = cfg.hidden_size
            self.budget_match = None
            if target is not None:
                count = (gru_parameter_count(cfg.hidden_size, self.n_in, self.n_pred, self.n_motor)
                         if cfg.budget_mode == "total" else gru_recurrent_weight_count(cfg.hidden_size))
                err = (count - target) / target
                if abs(err) > cfg.budget_tolerance:
                    raise BudgetMatchError(f"hidden_size={cfg.hidden_size} gives {count} {cfg.budget_mode} scalars, outside "
                                           f"+-{cfg.budget_tolerance:.0%} of the target {target}")
                self.budget_match = {"mode": cfg.budget_mode, "target": target, "achieved": count,
                                     "relative_error": err}
        elif target is not None:
            if cfg.budget_mode == "total":
                self.hidden_size, count = solve_hidden_size(self.n_in, self.n_pred, self.n_motor, target,
                                                            cfg.budget_tolerance, cfg.min_hidden)
            else:
                self.hidden_size, count = solve_hidden_size_recurrent(target, cfg.budget_tolerance, cfg.min_hidden)
            self.budget_match = {"mode": cfg.budget_mode, "target": target, "achieved": count,
                                 "relative_error": (count - target) / target}
        else:
            raise ValueError("give hidden_size or the target required by budget_mode")
        if cfg.prediction_horizon_frames < 1 or cfg.tbptt_frames < 1:
            raise ValueError("horizon and TBPTT length must be >= 1 frame")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.model = _GRUModel(self.n_in, self.hidden_size, self.n_pred, self.n_motor).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas,
                                    weight_decay=cfg.weight_decay)
        self._in_spec = sensory_stream_spec(encoder_cfg, cfg.frame_dt_s)
        self._out_spec = motor_stream_spec(encoder_cfg, cfg.frame_dt_s)
        self._frozen = False
        self._unit_mask = torch.ones(self.hidden_size, device=self.device)
        self._param_masks: dict[str, torch.Tensor] = {}
        self._lesion_log: list[dict] = []
        self._features = DenseStateFeatures(self)
        self._wall_s = 0.0
        self._grad_updates = 0
        self.reset()

    # ------------------------------------------------------------------ identity
    @property
    def metadata(self) -> SubstrateMetadata:
        caps = frozenset({Capability.GRADIENT_TRAINING, Capability.LESION, Capability.FREEZE,
                          Capability.EXACT_RESTORE})
        return SubstrateMetadata(
            kind="dense-rnn-gru", track="H2", seed=self.cfg.seed,
            config_hash=sha256_json({"dense": self.cfg.config_hash(), "encoder": self.encoder_cfg.config_hash(),
                                     "hidden_size": self.hidden_size}),
            input_spec_hash=self._in_spec.spec_hash(), version=DENSE_VERSION, capabilities=caps)

    def input_spec(self) -> StreamSpec:
        return self._in_spec

    def output_spec(self) -> StreamSpec:
        return self._out_spec

    # ------------------------------------------------------------------ dynamics
    def reset(self, seed: int | None = None) -> None:
        """Zero the hidden state and open a new episode. Parameters, ledger, and frame ids persist:
        frame ids are monotone over the substrate's lifetime so the exposure ledger stays valid."""
        if not hasattr(self, "_ledger"):
            self._ledger = IntervalLedger()
            self._replayed = 0
            self._frames_total = 0
            self._untrained_dropped = 0
            self._frame_id = -1                  # id of the latest frame stepped
            self._next_untrained = 0
        # inputs of the closing episode that were never trained are dropped (their targets, if any, stay unused)
        self._untrained_dropped += max(0, self._frame_id + 1 - max(self._next_untrained,
                                                                   getattr(self, "_window_origin", 0)))
        self.h = torch.zeros(1, self.hidden_size, device=self.device)
        self._buffer: dict[int, dict[str, torch.Tensor]] = {}
        self._window_origin = self._frame_id + 1  # no TBPTT window crosses an episode boundary
        self._next_untrained = max(self._next_untrained, self._window_origin)

    def _encode(self, frame: Mapping[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder.encode(frame)
        x = torch.as_tensor(self.encoder.vector(enc, include_efference=True), dtype=torch.float32,
                            device=self.device)
        s = torch.as_tensor(self.encoder.vector(enc, include_efference=False), dtype=torch.float32,
                            device=self.device)
        return x, s

    @torch.no_grad()
    def step(self, frame: Mapping[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
        if abs(dt - self.cfg.frame_dt_s) > 1e-9:
            raise ValueError(f"the dense comparator ticks at {self.cfg.frame_dt_s} s, got dt={dt}")
        problems = self._in_spec.validate(frame)
        if problems:
            raise ValueError("invalid frame: " + "; ".join(problems))
        t0 = time.perf_counter()
        x, s = self._encode(frame)
        self.h = self.model.cell(x[None], self.h) * self._unit_mask
        self._frame_id += 1
        self._frames_total += 1
        self._buffer[self._frame_id] = {"x": x, "s": s, "h": self.h.detach().clone()}
        motor = torch.sigmoid(self.model.motor(self.h))[0].reshape(self.encoder_cfg.n_joints, -1)
        self._trim()
        self._wall_s += time.perf_counter() - t0
        return {"motor_command": motor.cpu().numpy().astype(np.float64)}

    def _trim(self) -> None:
        """Keep one TBPTT window plus the horizon; inputs that fall out untrained are never trained."""
        keep_from = self._frame_id - (self.cfg.tbptt_frames + self.cfg.prediction_horizon_frames)
        for k in [k for k in self._buffer if k < keep_from]:
            del self._buffer[k]
        # an input is trainable while it and the hidden state before it are kept (at the episode
        # origin the hidden state before it is zero, so only the input itself must be kept)
        first_trainable = self._window_origin if self._window_origin >= keep_from else keep_from + 1
        if self._next_untrained < first_trainable:
            self._untrained_dropped += first_trainable - self._next_untrained
            self._next_untrained = first_trainable

    def expose_state(self) -> Mapping[str, np.ndarray]:
        return {"hidden": read_only(self.h[0].detach().cpu().numpy()),
                "frame_id": read_only(np.asarray(self._frame_id))}

    def features(self) -> DenseStateFeatures:
        return self._features

    # ------------------------------------------------------------------ learning (C2)
    def plasticity_step(self, modulators: Mapping[str, float] | None = None) -> PlasticityReport:
        raise CapabilityUnavailable(Capability.LOCAL_PLASTICITY, "the dense comparator learns only by gradients")

    def causal_batch(self) -> TrainBatch | None:
        """Every not-yet-trained input whose target has already been observed, up to one TBPTT window."""
        hzn = self.cfg.prediction_horizon_frames
        start = max(self._next_untrained, self._window_origin)
        last_input = min(self._frame_id - hzn, start + self.cfg.tbptt_frames - 1)
        if last_input < start:
            return None
        ids = range(start, last_input + 1)
        missing = [i for i in ids if i not in self._buffer or i + hzn not in self._buffer]
        if missing:
            raise RuntimeError(f"causal window frames were discarded: {missing[:5]}")
        if start == self._window_origin:
            h0 = torch.zeros(1, self.hidden_size, device=self.device)
        elif start - 1 in self._buffer:
            h0 = self._buffer[start - 1]["h"]
        else:
            raise RuntimeError(f"hidden state before frame {start} was discarded")
        return TrainBatch(start, torch.stack([self._buffer[i]["x"] for i in ids]),
                          torch.stack([self._buffer[i + hzn]["s"] for i in ids]), h0.detach())

    def _check_causal(self, batch: "TrainBatch") -> None:
        """An explicit batch must be a window of the causal buffer: same inputs, next-frame targets that were
        already observed, and the stored hidden state before it. Anything else is refused."""
        hzn = self.cfg.prediction_horizon_frames
        a, b = batch.target_ids
        if a < self._window_origin or b - 1 + hzn > self._frame_id:
            raise ValueError("batch reaches outside the current episode's observed frames")
        for i in range(a, b):
            if i not in self._buffer or i + hzn not in self._buffer:
                raise ValueError(f"frame {i} is no longer in the causal buffer")
            if not torch.equal(batch.inputs[i - a], self._buffer[i]["x"]) or \
                    not torch.equal(batch.targets[i - a], self._buffer[i + hzn]["s"]):
                raise ValueError(f"batch row {i - a} does not match the causal buffer")
        h0 = torch.zeros(1, self.hidden_size, device=self.device) if a == self._window_origin \
            else self._buffer.get(a - 1, {}).get("h")
        if h0 is None or not torch.equal(batch.h0, h0):
            raise ValueError("batch h0 is not the stored hidden state before its first input")

    def train_step(self, batch: Any = None) -> TrainReport:
        if self._frozen:
            raise SubstrateFrozenError("dense comparator is frozen")
        if batch is not None:
            self._check_causal(batch)
        batch = batch if batch is not None else self.causal_batch()
        if batch is None:
            return TrainReport(False, None, 0, {"reason": "no target observed yet"})
        a, b = batch.target_ids
        seen = self._ledger.overlap(a, b)
        if seen and self._replayed + seen > self.cfg.replay_budget_targets:
            raise SingleExposureViolation(f"targets {a}..{b - 1} include {seen} already trained on; declared "
                                          f"replay budget {self.cfg.replay_budget_targets}, used {self._replayed}")
        t0 = time.perf_counter()
        self.model.train()
        h = batch.h0
        preds = []
        for x in batch.inputs:
            h = self.model.cell(x[None], h) * self._unit_mask
            preds.append(self.model.pred(h)[0])
        loss = torch.mean((torch.stack(preds) - batch.targets) ** 2)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        params = dict(self.model.named_parameters())
        for name, mask in self._param_masks.items():          # lesioned weights get no gradient or moments
            if params[name].grad is not None:
                params[name].grad.mul_(mask)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm))
        self.opt.step()
        self._apply_masks()
        self.model.eval()
        self._ledger.add(a, b)
        self._replayed += seen
        self._next_untrained = max(self._next_untrained, b)
        self._grad_updates += 1
        self._wall_s += time.perf_counter() - t0
        return TrainReport(True, float(loss.detach()), b - a, {"first_target_input": a, "grad_norm": grad_norm,
                                                                "replayed_targets": seen})

    def start_episode(self) -> None:
        """Begin a new, discontinuous stream (same as reset for this substrate)."""
        self.reset()

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------------------ damage
    def _apply_masks(self) -> None:
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for name, mask in self._param_masks.items():
                params[name].mul_(mask)

    def lesion(self, spec: LesionSpec) -> LesionRecord:
        what, where = spec.target.split(":", 1)
        rng = np.random.default_rng(spec.seed)
        H = self.hidden_size
        params = dict(self.model.named_parameters())
        if (what, where) == ("units", "hidden"):
            alive = torch.nonzero(self._unit_mask > 0).flatten().cpu().numpy()
            k = int(round(spec.fraction * H))
            if k > alive.size:
                raise ValueError("not enough surviving units")
            dead = torch.from_numpy(np.sort(rng.choice(alive, size=k, replace=False))).to(self.device)
            self._unit_mask[dead] = 0.0
            rows = torch.ones(3 * H, device=self.device)
            for g in range(3):
                rows[g * H + dead] = 0.0
            cols = self._unit_mask
            self._mask("cell.weight_ih", rows[:, None].expand_as(params["cell.weight_ih"]))
            self._mask("cell.weight_hh", rows[:, None] * cols[None, :])
            self._mask("cell.bias_ih", rows)
            self._mask("cell.bias_hh", rows)
            self._mask("pred.weight", cols[None, :].expand_as(params["pred.weight"]))
            self._mask("motor.weight", cols[None, :].expand_as(params["motor.weight"]))
            with torch.no_grad():
                self.h.mul_(self._unit_mask)
            removed, total = k, H
        elif (what, where) == ("weights", "recurrent"):
            w = params["cell.weight_hh"]
            current = self._param_masks.get("cell.weight_hh", torch.ones_like(w))
            alive = torch.nonzero(current.flatten() > 0).flatten().cpu().numpy()
            k = int(round(spec.fraction * w.numel()))
            if k > alive.size:
                raise ValueError("not enough surviving recurrent weights")
            dead = torch.from_numpy(np.sort(rng.choice(alive, size=k, replace=False))).to(self.device)
            m = torch.ones(w.numel(), device=self.device)
            m[dead] = 0.0
            self._mask("cell.weight_hh", m.reshape(w.shape))
            removed, total = k, w.numel()
        else:
            raise ValueError(f"dense lesions target 'units:hidden' or 'weights:recurrent', not {spec.target!r}")
        self._apply_masks()
        rec = LesionRecord(spec.spec_hash(), spec.target, spec.fraction, removed, total, {})
        self._lesion_log.append(asdict(rec))
        return rec

    def _mask(self, name: str, mask: torch.Tensor) -> None:
        prev = self._param_masks.get(name)
        m = mask.detach().to(self.device).contiguous().clone()
        self._param_masks[name] = m if prev is None else prev * m

    # ------------------------------------------------------------------ budget, persistence
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def budget(self) -> BudgetReport:
        n = self.parameter_count()
        return BudgetReport(
            kind="dense-rnn-gru", learned_scalars=n, active_synapses=None, parameter_count=n,
            frames=self._frames_total, simulated_s=self._frames_total * self.cfg.frame_dt_s, wall_s=self._wall_s,
            peak_memory_bytes=peak_rss_bytes(self.device), device=str(self.device),
            gradient_updates=self._grad_updates,
            notes={"hidden_size": self.hidden_size, "budget_match": self.budget_match,
                   "recurrent_weight_count": gru_recurrent_weight_count(self.hidden_size),
                   "targets_trained": self._ledger.total(), "replayed_targets": self._replayed,
                   "untrained_inputs_dropped": self._untrained_dropped,
                   "torch": torch.__version__,
                   "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"})

    def state_dict(self) -> dict:
        return {"format": DENSE_VERSION, "metadata": self.metadata.as_dict(),
                "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                # deep copy: Optimizer.state_dict() returns live references to its moment tensors,
                # and load_state_dict keeps them if dtype/device match, which would alias two substrates
                "optimizer": copy.deepcopy(self.opt.state_dict()),
                "h": self.h.detach().cpu().clone(), "frame_id": self._frame_id,
                "buffer": {k: {n: t.cpu().clone() for n, t in v.items()} for k, v in self._buffer.items()},
                "next_untrained": self._next_untrained, "window_origin": self._window_origin,
                "ledger": list(self._ledger.intervals), "replayed": self._replayed,
                "frames_total": self._frames_total, "untrained_dropped": self._untrained_dropped,
                "grad_updates": self._grad_updates, "frozen": self._frozen,
                "unit_mask": self._unit_mask.cpu().clone(),
                "param_masks": {k: v.cpu().clone() for k, v in self._param_masks.items()},
                "lesion_log": [dict(x) for x in self._lesion_log]}

    def load_state_dict(self, state: dict) -> None:
        if state.get("format") != DENSE_VERSION:
            raise ValueError(f"not a {DENSE_VERSION} state")
        if state["metadata"]["config_hash"] != self.metadata.config_hash:
            raise ValueError("state was saved from a differently configured dense substrate")
        d = self.device
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.h = state["h"].to(d).clone()
        self._frame_id = int(state["frame_id"])
        self._buffer = {int(k): {n: t.to(d).clone() for n, t in v.items()} for k, v in state["buffer"].items()}
        self._next_untrained = int(state["next_untrained"])
        self._window_origin = int(state["window_origin"])
        self._ledger = IntervalLedger([tuple(x) for x in state["ledger"]])
        self._replayed = int(state["replayed"])
        self._frames_total = int(state["frames_total"])
        self._untrained_dropped = int(state["untrained_dropped"])
        self._grad_updates = int(state["grad_updates"])
        self._frozen = bool(state["frozen"])
        self._unit_mask = state["unit_mask"].to(d).clone()
        self._param_masks = {k: v.to(d).clone() for k, v in state["param_masks"].items()}
        self._lesion_log = [dict(x) for x in state["lesion_log"]]

    def state_hash(self) -> str:
        s = self.state_dict()
        s.pop("optimizer")        # hashed separately: its nested layout is torch-version dependent
        return sha256_json({"state": state_digest(s), "optimizer": state_digest(self.opt.state_dict()["state"])})
