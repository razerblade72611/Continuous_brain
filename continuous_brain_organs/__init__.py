"""Organs of the Continuous Brain individual: hosts that turn raw sensor data into perceived state.

Each organ is a fixed function (weights pinned by SHA-256 against a model card) whose output the individual
writes to its canonical store every tick.  See docs/explore/VISION_ORGAN.md.
"""
from .vision import (MATERIALS, OBJECTS, GRID_COLS, GRID_ROWS, PLAYER_COL, PLAYER_ROW, TileDecoder, VisionOrgan,
                     WeightsMismatch)

__all__ = ["MATERIALS", "OBJECTS", "GRID_ROWS", "GRID_COLS", "PLAYER_ROW", "PLAYER_COL", "TileDecoder",
           "VisionOrgan", "WeightsMismatch"]
