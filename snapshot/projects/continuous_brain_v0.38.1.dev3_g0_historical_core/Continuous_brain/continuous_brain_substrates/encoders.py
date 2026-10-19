"""Track-neutral sensory encoders.

Amendment A1.1 requires the dense comparator to receive "the same input
channels and encoders as DEV, fed as continuous values". So the encoder lives
here, outside both tracks:

  retina         difference-of-Gaussians centre-surround filter over the
                 luminance frame; ON = positive part, OFF = negative part
  proprioception Gaussian population codes over each cell's preferred joint
                 angle / angular velocity (centres evenly spaced over the
                 physical range, width 1.5 x spacing)
  efference      the motor command issued at this frame, passed through

The SNN adapter scales these into input currents for the matching sensory
populations; the dense comparator reads the same vector as continuous input.
Everything is float64 numpy, deterministic, and has no learned parameters.

Luminance frames are indexed [y, x]. Encoded retina vectors are flattened with
x as the slow index, which matches the E0 anatomy's cell order
(`continuous_brain_snn.anatomy.build_anatomy`: idx = gx * side + gy).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from continuous_brain_protocols import ChannelKind, ChannelSpec, StreamSpec, sha256_json


@dataclass(frozen=True)
class EncoderConfig:
    retina_side: int = 32
    dog_sigma_center: float = 0.8
    dog_sigma_surround: float = 2.0
    dog_radius: int = 5
    n_joints: int = 2
    angle_cells: int = 32
    velocity_cells: int = 32
    angle_range_rad: tuple[float, float] = (-1.5708, 1.5708)
    velocity_range_rad_s: tuple[float, float] = (-6.0, 6.0)
    tuning_width_factor: float = 1.5
    motor_channels: int = 3          # plus, minus, co-contraction

    def config_hash(self) -> str:
        return sha256_json(asdict(self))

    @classmethod
    def from_anatomy_config(cls, a) -> "EncoderConfig":
        """Mirror an E0 AnatomyConfig (duck-typed; this module does not import the SNN)."""
        return cls(retina_side=a.retina_side, n_joints=a.n_joints, angle_cells=a.proprio_angle_cells,
                   velocity_cells=a.proprio_velocity_cells, angle_range_rad=tuple(a.angle_range_rad),
                   velocity_range_rad_s=tuple(a.velocity_range_rad_s))


def dog_kernel(sigma_c: float, sigma_s: float, radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(r, r, indexing="ij")
    d2 = xx ** 2 + yy ** 2
    c = np.exp(-d2 / (2 * sigma_c ** 2))
    c /= c.sum()
    s = np.exp(-d2 / (2 * sigma_s ** 2))
    s /= s.sum()
    return c - s


def correlate_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D cross-correlation, zero padding, output the size of the input."""
    k = kernel.shape[0]
    r = k // 2
    h, w = image.shape
    padded = np.zeros((h + 2 * r, w + 2 * r), dtype=np.float64)
    padded[r:r + h, r:r + w] = image
    out = np.zeros((h, w), dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            out += kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]
    return out


def sensory_stream_spec(cfg: EncoderConfig, dt_s: float) -> StreamSpec:
    """The physical input channels every substrate consumes."""
    j = cfg.n_joints
    return StreamSpec(channels=(
        ChannelSpec("luminance", (cfg.retina_side, cfg.retina_side), "luminance", ChannelKind.SENSOR),
        ChannelSpec("joint_angle", (j,), "rad", ChannelKind.SENSOR, *cfg.angle_range_rad),
        ChannelSpec("joint_velocity", (j,), "rad/s", ChannelKind.SENSOR, *cfg.velocity_range_rad_s),
        ChannelSpec("motor_efference", (j, cfg.motor_channels), "normalized_command", ChannelKind.EFFERENCE,
                    0.0, 1.0),
    ), dt_s=dt_s)


def motor_stream_spec(cfg: EncoderConfig, dt_s: float) -> StreamSpec:
    return StreamSpec(channels=(
        ChannelSpec("motor_command", (cfg.n_joints, cfg.motor_channels), "normalized_command",
                    ChannelKind.MOTOR, 0.0, 1.0),
    ), dt_s=dt_s)


class SensoryEncoder:
    """Physical frame -> encoded continuous vectors, one block per sensory population."""

    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.kernel = dog_kernel(cfg.dog_sigma_center, cfg.dog_sigma_surround, cfg.dog_radius)
        self.angle_centres = np.linspace(*cfg.angle_range_rad, cfg.angle_cells)
        self.velocity_centres = np.linspace(*cfg.velocity_range_rad_s, cfg.velocity_cells)
        self.angle_width = self._width(cfg.angle_range_rad, cfg.angle_cells)
        self.velocity_width = self._width(cfg.velocity_range_rad_s, cfg.velocity_cells)

    def _width(self, rng: tuple[float, float], n: int) -> float:
        return (rng[1] - rng[0]) / max(n - 1, 1) * self.cfg.tuning_width_factor

    # Block layout, in order. The dense comparator's input vector is their concatenation.
    def blocks(self) -> list[tuple[str, int]]:
        c = self.cfg
        out = [("retina_on", c.retina_side ** 2), ("retina_off", c.retina_side ** 2)]
        for j in range(c.n_joints):
            out += [(f"proprio_angle_j{j}", c.angle_cells), (f"proprio_velocity_j{j}", c.velocity_cells)]
        out.append(("efference", c.n_joints * c.motor_channels))
        return out

    def sensory_size(self) -> int:
        return sum(n for name, n in self.blocks() if name != "efference")

    def input_size(self) -> int:
        return sum(n for _, n in self.blocks())

    def encode(self, frame) -> dict[str, np.ndarray]:
        lum = np.asarray(frame["luminance"], dtype=np.float64)
        dog = correlate_same(lum, self.kernel).T.reshape(-1)       # x as the slow index
        out = {"retina_on": np.maximum(dog, 0.0), "retina_off": np.maximum(-dog, 0.0)}
        angle = np.asarray(frame["joint_angle"], dtype=np.float64)
        vel = np.asarray(frame["joint_velocity"], dtype=np.float64)
        for j in range(self.cfg.n_joints):
            out[f"proprio_angle_j{j}"] = np.exp(-0.5 * ((self.angle_centres - angle[j]) / self.angle_width) ** 2)
            out[f"proprio_velocity_j{j}"] = np.exp(
                -0.5 * ((self.velocity_centres - vel[j]) / self.velocity_width) ** 2)
        out["efference"] = np.asarray(frame["motor_efference"], dtype=np.float64).reshape(-1)
        return out

    def vector(self, encoded: dict[str, np.ndarray], include_efference: bool = True) -> np.ndarray:
        parts = [encoded[name] for name, _ in self.blocks() if include_efference or name != "efference"]
        return np.concatenate(parts)
