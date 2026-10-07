"""Training batches from the archive: sequences of what was sensed, done and felt, with the targets the
predictive core and the skill policy learn from.

``StreamIndex`` loads one stream's small columns (everything but the frames) into memory and computes per-row
targets that need the future: the 12 risk labels (died / ran out of water / ran out of food / lost 3+ health,
within 25, 100 and 300 steps, inside the same life).  Frames are decoded a row group at a time and cached.

``SequenceSampler`` draws windows of ``T`` consecutive transitions of one stream (they may cross lives;
``is_first`` marks each life's first step), from the streams it was given, so a loader is built from exactly the
provenance it means to train on.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from continuous_brain_world import daylight as dl

from .archive import FEATURES, INVENTORY, Archive, ShardRef, fixed, frames

N_ACTIONS = 17
RISK_H = (25, 100, 300)
RISKS = ("die", "no_water", "no_food", "hurt3", "no_energy")   # the old foresight's five, at three horizons
N_RISK = len(RISKS) * len(RISK_H)
BODY_DIM = len(INVENTORY) + 1 + 2 + 2 + 2          # inventory, asleep, facing, daylight (sin, cos), position/64
MAX_STEPS = 10000


def body_vector(inv, sleeping, facing, step, pos) -> np.ndarray:
    """The body sense as numbers: what the core and the policy feel of their own body each step."""
    inv = np.asarray(inv, np.float32).reshape(-1, len(INVENTORY))
    t = (np.asarray(step, np.float64).reshape(-1) % dl.DAY_LENGTH) / dl.DAY_LENGTH
    return np.concatenate([inv, np.asarray(sleeping, np.float32).reshape(-1, 1),
                           np.asarray(facing, np.float32).reshape(-1, 2),
                           np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], 1).astype(np.float32),
                           np.asarray(pos, np.float32).reshape(-1, 2) / 64.0], 1)


def _next_true(flag: np.ndarray) -> np.ndarray:
    """For each i, the smallest j > i with flag[j] (len(flag) + big if none)."""
    n = len(flag)
    out = np.full(n, n + 10 ** 6, np.int64)
    nxt = n + 10 ** 6
    for i in range(n - 1, -1, -1):
        out[i] = nxt
        if flag[i]:
            nxt = i
    return out


def risk_targets(life: np.ndarray, step: np.ndarray, done: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """float32[N, 15]: for each row, what followed within each horizon, inside its own life (``RISKS`` x ``RISK_H``,
    risk-major)."""
    n = len(life)
    out = np.zeros((n, N_RISK), np.float32)
    h, food, drink, energy = inv[:, 0].astype(np.int64), inv[:, 1], inv[:, 2], inv[:, 3]
    starts = np.flatnonzero(np.r_[True, (life[1:] != life[:-1]) | (step[1:] <= step[:-1])])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        L = e - s
        idx = np.arange(L)
        died = bool(done[e - 1]) and int(step[e - 1]) + 1 < MAX_STEPS
        until_end = (L - 1) - idx + 1               # steps from row i until the life ends
        nw, nf = _next_true(drink[s:e] == 0) - idx, _next_true(food[s:e] == 0) - idx
        ne = _next_true(energy[s:e] == 0) - idx
        pad = np.r_[h[s:e], np.full(max(RISK_H), 10 ** 6)]
        for j, H in enumerate(RISK_H):
            win = np.lib.stride_tricks.sliding_window_view(pad[1:], H)[:L]       # rows i+1 .. i+H
            out[s:e, 0 * 3 + j] = died & (until_end <= H)
            out[s:e, 1 * 3 + j] = nw <= H
            out[s:e, 2 * 3 + j] = nf <= H
            out[s:e, 3 * 3 + j] = win.min(1) <= h[s:e] - 3
            out[s:e, 4 * 3 + j] = ne <= H
    return out


@dataclass
class StreamIndex:
    stream: str
    where: str
    shards: list[ShardRef]
    offsets: np.ndarray          # first stream row of each shard
    cols: dict                   # small per-row arrays
    namespaces: tuple

    @classmethod
    def load(cls, archive: Archive, refs: list[ShardRef], namespaces=("student",)) -> "StreamIndex":
        small = ["life", "step", "done", "action", "reward", "reward_caretaker", "inventory", "pos", "facing",
                 "sleeping", "who"]
        parts = {k: [] for k in small}
        extra = {"teacher": ["teacher_action", "teacher_option"], "organ": ["features"]}
        for ns in namespaces:
            for k in extra.get(ns, []):
                parts[k] = []
        offsets, n = [], 0
        for ref in refs:
            offsets.append(n)
            t = archive.read(ref, ("student",), {"student": small})["student"]
            for k in ("life", "step", "done", "action", "reward", "reward_caretaker", "sleeping", "who"):
                parts[k].append(t[k].to_numpy(zero_copy_only=False))
            parts["inventory"].append(fixed(t, "inventory", np.int16))
            parts["pos"].append(fixed(t, "pos", np.int16))
            parts["facing"].append(fixed(t, "facing", np.int16))
            if "teacher" in namespaces:
                tt = archive.read(ref, ("teacher",), {"teacher": extra["teacher"]})["teacher"]
                for k in extra["teacher"]:
                    parts[k].append(tt[k].to_numpy(zero_copy_only=False))
            if "organ" in namespaces:
                to = archive.read(ref, ("organ",), {"organ": ["features"]})["organ"]
                parts["features"].append(fixed(to, "features", np.float32))
            n += ref.manifest["rows"]
        cols = {k: np.concatenate(v) for k, v in parts.items()}
        cols["risk"] = risk_targets(cols["life"], cols["step"], cols["done"], cols["inventory"])
        cols["is_first"] = np.r_[True, (cols["life"][1:] != cols["life"][:-1]) | (cols["step"][1:] <= cols["step"][:-1])]
        cols["body"] = body_vector(cols["inventory"], cols["sleeping"], cols["facing"], cols["step"], cols["pos"])
        return cls(refs[0].stream, refs[0].where, refs, np.asarray(offsets), cols, tuple(namespaces))

    def __len__(self):
        return len(self.cols["life"])


class FrameCache:
    def __init__(self, archive: Archive, capacity: int = 16):
        self.archive, self.capacity = archive, capacity
        self.cache: OrderedDict = OrderedDict()

    def group(self, ref: ShardRef, g: int, ns: str = "student", column: str = "frame") -> np.ndarray:
        key = (ref.stream, ref.manifest["shard"], ns, column, g)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        t = self.archive.read_group(ref, ns, g, columns=[column])
        arr = frames(t) if column == "frame" else fixed(t, column, np.int8)
        self.cache[key] = arr
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return arr


class SequenceSampler:
    """Random windows of T steps.  ``batch()`` returns numpy arrays shaped [B, T, ...]."""

    def __init__(self, archive: Archive, refs: list[ShardRef], T: int = 64, seed: int = 0,
                 namespaces=("student",), row_group: int = 4096, cache_groups: int = 24, map_crop: bool = False):
        self.archive, self.T = archive, T
        by_stream: dict[str, list[ShardRef]] = {}
        for r in refs:
            by_stream.setdefault(r.stream, []).append(r)
        self.streams = [StreamIndex.load(archive, sorted(v, key=lambda r: r.manifest["shard"]), namespaces)
                        for v in by_stream.values()]
        self.streams = [s for s in self.streams if len(s) >= T]
        if not self.streams:
            raise ValueError(f"no stream has {T} rows")
        w = np.asarray([len(s) - T + 1 for s in self.streams], np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self.row_group = row_group
        self.cache = FrameCache(archive, cache_groups)
        self.map_crop = map_crop

    def rows(self) -> int:
        return sum(len(s) for s in self.streams)

    def _column(self, s: StreamIndex, start: int, T: int, ns: str, column: str) -> np.ndarray:
        out = []
        i, end = start, start + T
        while i < end:
            k = int(np.searchsorted(s.offsets, i, side="right") - 1)
            ref, local = s.shards[k], i - s.offsets[k]
            g = local // self.row_group
            arr = self.cache.group(ref, g, ns, column)
            a0 = local - g * self.row_group
            take = min(end - i, len(arr) - a0)
            out.append(arr[a0:a0 + take])
            i += take
        return np.concatenate(out)

    def window(self, s: StreamIndex, start: int) -> dict:
        T, c = self.T, s.cols
        sl = slice(start, start + T)
        act = c["action"][sl].astype(np.int64)
        prev = np.r_[0, act[:-1]]
        rew = (c["reward"] + c["reward_caretaker"]).astype(np.float32)
        prev_rew = np.r_[0.0, rew[start:start + T - 1]] if start == 0 else rew[start - 1:start + T - 1]
        is_first = c["is_first"][sl].copy()
        is_first[0] = True                                  # every window starts its own recurrent state
        out = {"image": self._column(s, start, T, "student", "frame"), "body": c["body"][sl],
               "action": act, "prev_action": np.where(is_first, 0, prev), "reward": np.where(is_first, 0.0, prev_rew),
               "cont": 1.0 - c["done"][sl].astype(np.float32), "is_first": is_first.astype(np.float32),
               "risk": c["risk"][sl], "who": c["who"][sl]}
        if "teacher_action" in c:
            out["teacher_action"] = c["teacher_action"][sl].astype(np.int64)
            out["teacher_option"] = c["teacher_option"][sl].astype(np.int64)
        if "features" in c:
            out["features"] = c["features"][sl]
        if self.map_crop:
            out["map_crop"] = self._column(s, start, T, "organ", "map_crop")
        return out

    def batch(self, B: int) -> dict:
        ws = []
        for _ in range(B):
            s = self.streams[int(self.rng.choice(len(self.streams), p=self.p))]
            ws.append(self.window(s, int(self.rng.integers(0, len(s) - self.T + 1))))
        return {k: np.stack([w[k] for w in ws]) for k in ws[0]}
