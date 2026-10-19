"""Small, exact learning machinery for the individual's learned faculties (exploration lane).

Everything a learner needs between ticks -- weights, the target copy, Adam's moments, its replay memory -- is a
plain float32 vector laid out by a ``Layout``, so it lives in one canonical-store channel and is saved, restored,
hashed and erased like every other piece of the individual's state.  The networks are numpy MLPs with ReLU
hidden layers and hand-written backpropagation (checked against finite differences in the tests), so a learning
step is a pure, deterministic function of the stored state and the experience.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

F = np.float32


# ---------------------------------------------------------------------------------------------------- layout
class Layout:
    """Named float32 arrays packed into one flat vector."""

    def __init__(self, parts: list[tuple[str, tuple[int, ...]]]):
        self.parts, self.slices, off = [], {}, 0
        for name, shape in parts:
            n = int(np.prod(shape)) if shape else 1
            self.parts.append((name, tuple(shape)))
            self.slices[name] = (off, off + n, tuple(shape))
            off += n
        self.size = off

    def view(self, flat: np.ndarray, name: str) -> np.ndarray:
        a, b, shape = self.slices[name]
        return flat[a:b].reshape(shape)

    def views(self, flat: np.ndarray) -> dict:
        return {n: self.view(flat, n) for n, _ in self.parts}

    def zeros(self) -> np.ndarray:
        return np.zeros(self.size, F)


# --------------------------------------------------------------------------------------------------------- MLP
def mlp_shapes(sizes: list[int]) -> list[tuple[str, tuple[int, ...]]]:
    out = []
    for i in range(len(sizes) - 1):
        out += [(f"W{i}", (sizes[i], sizes[i + 1])), (f"b{i}", (sizes[i + 1],))]
    return out


def mlp_init(rng: np.random.Generator, sizes: list[int], out_scale: float = 0.1) -> dict:
    p = {}
    for i in range(len(sizes) - 1):
        fan_in = sizes[i]
        scale = np.sqrt(2.0 / fan_in) * (out_scale if i == len(sizes) - 2 else 1.0)
        p[f"W{i}"] = (rng.standard_normal((sizes[i], sizes[i + 1])) * scale).astype(F)
        p[f"b{i}"] = np.zeros(sizes[i + 1], F)
    return p


def mlp_forward(p: dict, x: np.ndarray, n_layers: int):
    """Returns (output, cache).  ReLU between layers, linear output."""
    h, cache = x.astype(F), []
    for i in range(n_layers):
        z = h @ p[f"W{i}"] + p[f"b{i}"]
        cache.append((h, z))
        h = np.maximum(z, 0) if i < n_layers - 1 else z
    return h, cache


def mlp_backward(p: dict, cache, dout: np.ndarray, n_layers: int) -> dict:
    g, d = {}, dout.astype(F)
    for i in reversed(range(n_layers)):
        h, z = cache[i]
        if i < n_layers - 1:
            d = d * (z > 0)
        g[f"W{i}"] = h.T @ d
        g[f"b{i}"] = d.sum(0)
        d = d @ p[f"W{i}"].T
    return g


@dataclass
class Adam:
    lr: float = 3e-4
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    clip: float = 10.0

    def step(self, params: dict, grads: dict, m: dict, v: dict, t: int) -> None:
        """In place on the views (params, m, v are views into a fresh copy of the store vector)."""
        norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-12))
        for k, g in grads.items():
            g = g * scale
            m[k][...] = self.b1 * m[k] + (1 - self.b1) * g
            v[k][...] = self.b2 * v[k] + (1 - self.b2) * g * g
            mh = m[k] / (1 - self.b1 ** t)
            vh = v[k] / (1 - self.b2 ** t)
            params[k][...] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)


def huber_grad(err: np.ndarray, delta: float = 1.0) -> tuple[float, np.ndarray]:
    a = np.abs(err)
    loss = np.where(a <= delta, 0.5 * err * err, delta * (a - 0.5 * delta))
    grad = np.where(a <= delta, err, delta * np.sign(err))
    return float(loss.mean()), grad / err.size


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def bce_grad(logits: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    p = sigmoid(logits)
    w = np.ones_like(y) if w is None else w
    loss = -(w * (y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))).sum() / max(1.0, float(w.sum()))
    return float(loss), (w * (p - y) / max(1.0, float(w.sum()))).astype(F)


# ------------------------------------------------------------------------------------------------- replay ring
class Ring:
    """A fixed-capacity ring of float32 rows inside a store vector (a view), with count and head in the caller's
    bookkeeping state."""

    @staticmethod
    def push(buf: np.ndarray, head: int, count: int, row: np.ndarray) -> tuple[int, int]:
        buf[head] = row
        cap = buf.shape[0]
        return (head + 1) % cap, min(cap, count + 1)

    @staticmethod
    def sample(rng: np.random.Generator, count: int, n: int) -> np.ndarray:
        return rng.integers(0, count, size=n)
