"""The Crafter base world, made deterministic and snapshot-able.

``DeterministicCrafter`` is ``crafter.Env`` (crafter 1.8.3, unmodified on disk) with exactly one change:

**Determinism fix.**  Crafter's ``_balance_object`` despawns ``creatures[random.randint(0, len(creatures))]``
where ``creatures`` is built by iterating a Python ``set`` of objects.  Objects hash by ``id()`` (memory
address), so the set order -- and therefore *which* zombie/cow/skeleton is despawned for the same random
draw -- varies between processes and after an unpickle.  Measured here: the same seed and the same action
sequence gave four different trajectories in four processes (different death steps).  The override sorts the
creatures by their index in ``world._objects`` (stable, preserved by pickling) before the same random draw.
The despawn rule stays a uniform choice among the same creatures with the same probabilities; only the
arbitrary order is fixed.  Every other line of the function is Crafter's own (MIT licence, Danijar Hafner).

Other facts this module relies on (read from crafter's source):

* ``reset`` seeds the world with ``hash((seed, episode)) % (2**31 - 1)``; tuples of ints hash
  deterministically, and ``CrafterBridge.reset`` resets the episode counter so a W1 episode depends only on
  its seed.
* The night renderer draws its noise from ``world.random`` -- the same RNG that drives world dynamics.  So the
  primary observation's size is part of the dynamics (it fixes how many noise numbers are drawn), and any
  *extra* render (``render(size)``) is wrapped to save and restore the RNG state so it has no side effect.
* Snapshots pickle the whole env except the texture cache (``persistent_id``), so world, player, creatures,
  RNG state, chunk bookkeeping and step counter round-trip bit-exactly.
"""
from __future__ import annotations

import hashlib
import io
import pickle
from importlib import metadata
from pathlib import Path

import crafter
import numpy as np
from crafter import constants, engine, objects

_TEXTURE_PID = "w1-crafter-textures"


def crafter_provenance() -> dict:
    root = Path(crafter.__file__).parent
    return {"crafter_version": metadata.version("crafter"),
            "data_yaml_sha256": hashlib.sha256((root / "data.yaml").read_bytes()).hexdigest(),
            "objects_py_sha256": hashlib.sha256((root / "objects.py").read_bytes()).hexdigest(),
            "env_py_sha256": hashlib.sha256((root / "env.py").read_bytes()).hexdigest()}


class DeterministicCrafter(crafter.Env):
    """``crafter.Env`` with a deterministic despawn order (see module docstring)."""

    def _balance_object(self, chunk, objs, cls, material, span_dist, despan_dist,
                        spawn_prob, despawn_prob, ctor, target_fn):
        xmin, xmax, ymin, ymax = chunk
        random = self._world.random
        obj_map = self._world._obj_map
        creatures = sorted((obj for obj in objs if isinstance(obj, cls)),
                           key=lambda o: int(obj_map[o.pos[0], o.pos[1]]))
        mask = self._world.mask(*chunk, material)
        target_min, target_max = target_fn(len(creatures), mask.sum())
        if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
            xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
            ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
            xs, ys = xs[mask], ys[mask]
            i = random.randint(0, len(xs))
            pos = np.array((xs[i], ys[i]))
            empty = self._world[pos][1] is None
            away = self._player.distance(pos) >= span_dist
            if empty and away:
                self._world.add(ctor(pos))
        elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
            obj = creatures[random.randint(0, len(creatures))]
            away = self._player.distance(obj.pos) >= despan_dist
            if away:
                self._world.remove(obj)


class _Pickler(pickle.Pickler):
    def __init__(self, file, textures):
        super().__init__(file, protocol=5)
        self._textures = textures

    def persistent_id(self, obj):
        return _TEXTURE_PID if obj is self._textures else None


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, textures):
        super().__init__(file)
        self._textures = textures

    def persistent_load(self, pid):
        if pid != _TEXTURE_PID:
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")
        return self._textures


MATERIAL_IDS = None  # filled on first bridge construction (crafter assigns ids at World construction)


class CrafterBridge:
    """Owns one ``DeterministicCrafter``; resets it per seed, steps it, snapshots and restores it."""

    def __init__(self, obs_size: int = 64, length: int = 10000):
        global MATERIAL_IDS
        self.obs_size = int(obs_size)
        self.length = int(length)
        self.env = DeterministicCrafter(area=(64, 64), view=(9, 9), size=(self.obs_size, self.obs_size),
                                        reward=True, length=self.length, seed=0)
        self._textures = self.env._textures
        MATERIAL_IDS = dict(self.env._world._mat_ids)
        self.last_crafter_reward = 0.0

    # -- accessors ---------------------------------------------------------------------------------------
    @property
    def world(self) -> engine.World:
        return self.env._world

    @property
    def player(self) -> objects.Player:
        return self.env._player

    @property
    def step_count(self) -> int:
        return int(self.env._step)

    @property
    def unit(self) -> int:
        return int((np.array([self.obs_size, self.obs_size]) // self.env._view)[0])

    @property
    def view_grid(self) -> tuple[int, int]:
        g = self.env._local_view._grid
        return int(g[0]), int(g[1])

    # -- dynamics -----------------------------------------------------------------------------------------
    def reset(self, seed: int) -> np.ndarray:
        self.env._seed = int(seed)
        self.env._episode = 0
        return self.env.reset()

    def step(self, action: int):
        obs, reward, done, info = self.env.step(int(action))
        self.last_crafter_reward = float(reward)
        return obs, float(reward), bool(done)

    def render(self, size: int) -> np.ndarray:
        """Render at another resolution without touching world dynamics (RNG state saved and restored)."""
        state = self.world.random.get_state()
        try:
            return self.env.render((int(size), int(size)))
        finally:
            self.world.random.set_state(state)

    def semantic(self) -> np.ndarray:
        return self.env._sem_view()

    # -- persistence --------------------------------------------------------------------------------------
    def snapshot_bytes(self) -> bytes:
        buf = io.BytesIO()
        _Pickler(buf, self._textures).dump(self.env)
        return buf.getvalue()

    def restore_bytes(self, data: bytes) -> None:
        env = _Unpickler(io.BytesIO(data), self._textures).load()
        if not isinstance(env, DeterministicCrafter):
            raise TypeError("snapshot does not hold a DeterministicCrafter")
        if tuple(env._size) != (self.obs_size, self.obs_size):
            raise ValueError("snapshot was taken with a different observation size")
        self.env = env


def object_kind(obj) -> str:
    return {objects.Player: "player", objects.Cow: "cow", objects.Zombie: "zombie",
            objects.Skeleton: "skeleton", objects.Arrow: "arrow", objects.Plant: "sapling",
            objects.Fence: "fence"}.get(type(obj), type(obj).__name__.lower())


WALKABLE = tuple(constants.walkable)
