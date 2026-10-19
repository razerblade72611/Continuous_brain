"""Shared battery configuration and provenance (H1/H2 shared battery, C5).

Every battery result in v0.38.0.dev0 is ENGINEERING_ONLY with metric status
UNREGISTERED_METRIC: no item has a preregistered task definition, metric,
margin, seed discipline, or decision rule yet (ChatGPT's C5 clarification).
Seeds must come from an ENGINEERING family in registry/seed_ranges.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from continuous_brain_protocols import SeedPurpose, Substrate, sha256_json
from continuous_brain_snn.provenance import code_revision, environment

from ..results import ResultRecord
from ..seeds import SeedRegistry

SubstrateFactory = Callable[[int], Substrate]
SEED_SPAN = 2000

ENGINEERING_NOTE = ("engineering validation of the harness only: not comparative evidence for H1/H2; "
                    "no preregistered metric, margin, or decision rule")


@dataclass(frozen=True)
class BatteryConfig:
    seed: int = 900000
    seed_family: str = "v0.38-ENGINEERING"
    develop_frames: int = 1000
    eval_frames: int = 800
    washout_frames: int = 50
    horizon_frames: int = 10                       # 100 ms at 10 ms frames
    raw_lags: int = 20
    feature_cap: int = 2000
    b2_tasks: int = 3
    b3_continue_frames: int = 200
    b3_drift_windows: int = 4
    b4_positions: int = 4
    b4_delays_frames: tuple[int, ...] = (10, 30, 100)
    b4_cue_interval_s: tuple[float, float] = (1.5, 2.5)
    b4_eval_frames: int = 6000
    b5_period_s: float = 0.25
    b6_fraction: float = 0.10
    b6_recovery_frames: int = 1000
    lesion_targets: tuple[tuple[str, str], ...] = (("H1", "synapses:cortex"), ("H2", "weights:recurrent"))

    def config_hash(self) -> str:
        return sha256_json(asdict(self))

    def lesion_target(self, track: str) -> str:
        for t, target in self.lesion_targets:
            if t == track:
                return target
        raise KeyError(f"no lesion target configured for track {track}")


def check_engineering_seed(cfg: BatteryConfig, registry: SeedRegistry | None = None) -> None:
    reg = registry or SeedRegistry.load()
    fam = reg.get(cfg.seed_family)
    if fam.purpose != SeedPurpose.ENGINEERING:
        raise ValueError(f"battery smoke runs must use an ENGINEERING seed family, not {fam.purpose.value}")
    top = cfg.seed + SEED_SPAN                    # items derive seeds from seed .. seed + SEED_SPAN
    if not (fam.start <= cfg.seed and top <= fam.stop):
        raise ValueError(f"seeds {cfg.seed}..{top} are not inside {fam.name} [{fam.start}-{fam.stop}]")


def base_record(item: str, sub: Substrate, cfg: BatteryConfig, stream_hash: str, extra: dict | None = None) -> ResultRecord:
    from continuous_brain_protocols import ResultStatus
    meta = sub.metadata
    prov = {"config_hash": meta.config_hash, "battery_config_hash": cfg.config_hash(), "stream_config_hash": stream_hash,
            "code_revision": code_revision(), "environment": environment(),
            "seed_family": {"name": cfg.seed_family, "seed": cfg.seed},
            "substrate": meta.as_dict(), "budget": sub.budget().as_dict(), "checkpoint_sha256": None,
            "artifacts": {}}
    if extra:
        prov.update(extra)
    return ResultRecord(item=item, status=ResultStatus.ENGINEERING_ONLY,
                        metric_status=ResultStatus.UNREGISTERED_METRIC, substrate=meta.kind, provenance=prov,
                        notes=[ENGINEERING_NOTE])


@dataclass
class Timer:
    marks: dict[str, float] = field(default_factory=dict)
