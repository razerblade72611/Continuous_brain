"""Exact day/night arithmetic of Crafter's daylight cycle.

Crafter (``crafter.Env._update_time``) sets, after ``step`` environment steps,

    progress = (step / 300) % 1 + 0.3
    daylight = 1 - abs(cos(pi * progress)) ** 3

and its renderer darkens the frame and overlays noise exactly when ``daylight < 0.5``
(``crafter.engine.LocalView._light``).  W1 therefore calls a step *dark* when
``daylight(step) < 0.5`` and derives every caretaker time phrase from that one rule:

* the cycle has period ``DAY_LENGTH = 300`` steps (checked, not assumed: ``_verify_periodicity``);
* within a cycle (``t = step % 300``) the dark steps are ``t in [NIGHTFALL_T, DAWN_T)`` = ``[148, 273)``;
* the brightest step is ``t = NOON_T = 60`` (daylight == 1.0); the darkest is ``t = MIDNIGHT_T = 210``.

Days and phases (all half-open step intervals):

* day 0 is ``[0, 273)``; day ``d >= 1`` is ``[273 + 300 (d - 1), 273 + 300 d)``.  A day begins at dawn.
* phase ``morning``  = from dawn (or step 0) up to noon,     ``t in [273, 300) U [0, 60)``;
  phase ``afternoon`` = from noon up to nightfall,          ``t in [60, 148)``;
  phase ``night``     = the dark steps,                      ``t in [148, 273)``.

Caretaker phrases map to deadlines (the last step at which an obligation can still be met):

* "before nightfall" / "before it gets dark"  ->  ``next_nightfall(step) - 1``
* "by tomorrow morning" / "by morning"        ->  ``next_dawn(step)``
* "by noon"                                   ->  ``next_noon(step)``
"""
from __future__ import annotations

import numpy as np

DAY_LENGTH = 300
DARK_THRESHOLD = 0.5


def daylight(step: int) -> float:
    """Bit-identical replica of ``crafter.Env._update_time`` (same operations, same dtypes)."""
    progress = (step / 300) % 1 + 0.3
    return float(1 - np.abs(np.cos(np.pi * progress)) ** 3)


def is_dark(step: int) -> bool:
    return daylight(step) < DARK_THRESHOLD


_CYCLE = [daylight(t) for t in range(DAY_LENGTH)]
_DARK = [v < DARK_THRESHOLD for v in _CYCLE]
NIGHTFALL_T = next(t for t in range(1, DAY_LENGTH) if _DARK[t] and not _DARK[t - 1])
DAWN_T = next(t for t in range(1, DAY_LENGTH) if not _DARK[t] and _DARK[t - 1])
NOON_T = int(np.argmax(_CYCLE))
MIDNIGHT_T = int(np.argmin(_CYCLE))
assert (NIGHTFALL_T, DAWN_T, NOON_T, MIDNIGHT_T) == (148, 273, 60, 210), (NIGHTFALL_T, DAWN_T, NOON_T, MIDNIGHT_T)
assert all(_DARK[t] == (NIGHTFALL_T <= t < DAWN_T) for t in range(DAY_LENGTH))
# Smallest distance of any cycle value from the threshold: periodic arithmetic is exact as long as the float
# error of (step / 300) % 1 stays far below this (it is ~1e-12 at step 1e6; the margin is ~5e-3).
THRESHOLD_MARGIN = min(abs(v - DARK_THRESHOLD) for v in _CYCLE)


def _verify_periodicity(max_step: int) -> bool:
    """Direct evaluation agrees with the periodic table for every step below ``max_step`` (used by tests)."""
    return all(is_dark(s) == _DARK[s % DAY_LENGTH] for s in range(max_step))


def day_index(step: int) -> int:
    return 0 if step < DAWN_T else (step - DAWN_T) // DAY_LENGTH + 1


def day_start(day: int) -> int:
    return 0 if day == 0 else DAWN_T + DAY_LENGTH * (day - 1)


def phase(step: int) -> str:
    t = step % DAY_LENGTH
    if NIGHTFALL_T <= t < DAWN_T:
        return "night"
    if NOON_T <= t < NIGHTFALL_T:
        return "afternoon"
    return "morning"


def _next_with_offset(step: int, offset: int) -> int:
    """Smallest s > step with s % DAY_LENGTH == offset."""
    base = step - step % DAY_LENGTH + offset
    return base if base > step else base + DAY_LENGTH


def next_nightfall(step: int) -> int:
    """First dark step strictly after ``step``'s current light span (strictly greater than ``step``)."""
    return _next_with_offset(step, NIGHTFALL_T)


def next_dawn(step: int) -> int:
    return _next_with_offset(step, DAWN_T)


def next_noon(step: int) -> int:
    return _next_with_offset(step, NOON_T)


def morning_window(day: int) -> tuple[int, int]:
    """Half-open step interval of the morning of ``day``."""
    start = day_start(day)
    return start, (NOON_T if day == 0 else start + (DAY_LENGTH - DAWN_T) + NOON_T)


def describe(step: int) -> dict:
    return {"step": step, "day": day_index(step), "phase": phase(step), "daylight": daylight(step),
            "dark": is_dark(step)}
