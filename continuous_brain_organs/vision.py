"""The vision organ: Crafter/W1 pixels -> a perceived local tile map (material and object per tile).

The persistent individual perceives W1 from pixels only.  This organ maps the 64 x 64 x 3 uint8 frame to the
tiles of Crafter's local view.  It never reads the privileged material map or object list at run time; those are
used only by ``labels_from_privileged`` to make training/evaluation labels (a supervised developmental phase with
a teacher -- see docs/explore/VISION_ORGAN.md).

Geometry (derived from crafter 1.8.3's ``Env.render`` / ``LocalView``; recomputed by ``crafter_geometry`` and
tested pixel-for-pixel against the renderer)::

    unit        = size // view           = 64 // 9 = 7 px per tile
    item_rows   = ceil(16 items / 9)     = 2      (inventory strip)
    local grid  = 9 columns x (9 - 2) = 7 rows of world tiles
    border      = (64 - 7 * 9) // 2      = 0      (view drawn at the top-left; row 63 and column 63 stay black)
    frame[r, c] = canvas[c, r]           (Crafter draws x-major and transposes, so pixel rows are world y)

    world view  : frame[0:49, 0:63]      inventory strip: frame[49:63, 0:63]
    tile (row i, col j) covers frame[7 i : 7 i + 7, 7 j : 7 j + 7] and shows the world tile
        (x, y) = (player_x + j - 4, player_y + i - 3)

Conventions of every map this module returns (``material``, ``object``, the confidences, the labels):

* arrays are ``[GRID_ROWS, GRID_COLS] = [7, 9]``, indexed ``[row, col]``;
* row 0 is **north** (smallest world y; Crafter's ``move_up`` is ``(0, -1)``), row 6 is south;
* col 0 is **west** (smallest world x; ``move_left`` is ``(-1, 0)``), col 8 is east;
* the player stands at ``[PLAYER_ROW, PLAYER_COL] = [3, 4]`` (Crafter centres the view at offset grid // 2);
  the view is therefore asymmetric: 3 tiles north and south, 4 west and east.
* ``local_to_world(player_pos)`` gives the absolute (x, y) of every cell.

Label spaces (``MATERIALS`` index == crafter's own material id; ``OBJECTS`` follow crafter's object classes):

* material: ``outside`` (id 0 = crafter's ``None``: a cell beyond the 64 x 64 world edge, which the renderer
  leaves flat grey 127 -- it is part of the world's geometry, not darkness), water, grass, stone, path, sand,
  tree, lava, coal, iron, diamond, table, furnace.
* object: ``none``, ``player``, ``cow``, ``zombie``, ``skeleton``, ``arrow`` (any direction), ``sapling``
  (``objects.Plant`` with ``grown <= 300``, texture ``plant``), ``plant_ripe`` (``grown > 300``, texture
  ``plant-ripe``).  Crafter 1.8.3 also defines ``Fence``, but nothing can create one (no worldgen rule, no
  ``place`` recipe), so it is not a class; ``labels_from_privileged`` refuses an unknown object.

Darkness is not a class: at night the organ must still say what is on the tile (evaluation reports night
accuracy separately).  Each object is drawn with alpha over its tile's material, so both heads are defined on
every tile, including the player's tile.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------------------------------------------
# Label spaces
MATERIALS = ("outside", "water", "grass", "stone", "path", "sand", "tree", "lava", "coal", "iron", "diamond",
             "table", "furnace")
OBJECTS = ("none", "player", "cow", "zombie", "skeleton", "arrow", "sapling", "plant_ripe")
MATERIAL_INDEX = {name: i for i, name in enumerate(MATERIALS)}
OBJECT_INDEX = {name: i for i, name in enumerate(OBJECTS)}
# W1's object_kind() names -> object class (a Plant is "sapling" there; ripeness is a separate flag).
_KIND_TO_OBJECT = {"player": "player", "cow": "cow", "zombie": "zombie", "skeleton": "skeleton",
                   "arrow": "arrow", "sapling": "sapling"}


def crafter_geometry(size: int = 64, view=(9, 9), n_items: int | None = None) -> dict:
    """Recompute the pixel geometry of ``crafter.Env.render`` for a square frame of ``size`` pixels."""
    if n_items is None:
        from crafter import constants
        n_items = len(constants.items)
    view = np.array(view)
    unit = np.array([size, size]) // view
    item_rows = int(math.ceil(n_items / view[0]))
    cols, rows = int(view[0]), int(view[1] - item_rows)
    border = (np.array([size, size]) - unit * view) // 2
    return {"frame_size": int(size), "unit": int(unit[0]), "cols": cols, "rows": rows, "item_rows": item_rows,
            "border_col": int(border[0]), "border_row": int(border[1]),
            "view_px_rows": rows * int(unit[1]), "view_px_cols": cols * int(unit[0]),
            "player_row": rows // 2, "player_col": cols // 2}


GEOMETRY = crafter_geometry()
FRAME_SIZE = GEOMETRY["frame_size"]
UNIT = GEOMETRY["unit"]
GRID_ROWS, GRID_COLS = GEOMETRY["rows"], GEOMETRY["cols"]
PLAYER_ROW, PLAYER_COL = GEOMETRY["player_row"], GEOMETRY["player_col"]
VIEW_ROWS_PX, VIEW_COLS_PX = GEOMETRY["view_px_rows"], GEOMETRY["view_px_cols"]
_R0, _C0 = GEOMETRY["border_row"], GEOMETRY["border_col"]


def world_view_crop(frame: np.ndarray) -> np.ndarray:
    """The world-view part of a frame (``[49, 63, 3]``, or ``[B, 49, 63, 3]`` for a batch); a view, not a copy."""
    return frame[..., _R0:_R0 + VIEW_ROWS_PX, _C0:_C0 + VIEW_COLS_PX, :]


def tile_slice(row: int, col: int) -> tuple[slice, slice]:
    """Pixel rows/cols of local tile (row, col) in the full frame."""
    return (slice(_R0 + row * UNIT, _R0 + (row + 1) * UNIT), slice(_C0 + col * UNIT, _C0 + (col + 1) * UNIT))


def local_to_world(player_pos) -> tuple[np.ndarray, np.ndarray]:
    """Absolute world (x, y) of every local cell: two int arrays ``[GRID_ROWS, GRID_COLS]``."""
    px, py = int(player_pos[0]), int(player_pos[1])
    rows, cols = np.meshgrid(np.arange(GRID_ROWS), np.arange(GRID_COLS), indexing="ij")
    return px + cols - PLAYER_COL, py + rows - PLAYER_ROW


def labels_from_privileged(priv: dict) -> tuple[np.ndarray, np.ndarray]:
    """TEACHER labels for the current frame from W1's privileged ground truth (``info["privileged"]``).

    Returns ``(material int8[7, 9], object int8[7, 9])`` in the organ's local convention.  Uses
    ``material_map``, ``material_ids``, ``objects`` and ``player_pos``.  Never used at run time.
    """
    mat_map = np.asarray(priv["material_map"])
    ids = priv["material_ids"]
    expect = {(None if n == "outside" else n): i for i, n in enumerate(MATERIALS)}
    if dict(ids) != expect:
        raise ValueError(f"crafter material ids differ from the organ's label space: {ids}")
    xs, ys = local_to_world(priv["player_pos"])
    w, h = mat_map.shape
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    material = np.zeros((GRID_ROWS, GRID_COLS), np.int8)
    material[inside] = mat_map[xs[inside], ys[inside]].astype(np.int8)
    obj = np.zeros((GRID_ROWS, GRID_COLS), np.int8)
    px, py = priv["player_pos"]
    for o in priv["objects"]:
        kind = o["kind"]
        if kind not in _KIND_TO_OBJECT:
            raise ValueError(f"object kind {kind!r} is not in the organ's label space")
        c = int(o["pos"][0]) - int(px) + PLAYER_COL
        r = int(o["pos"][1]) - int(py) + PLAYER_ROW
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            name = _KIND_TO_OBJECT[kind]
            if name == "sapling" and o["ripe"]:
                name = "plant_ripe"
            if obj[r, c] != 0:
                raise ValueError(f"two objects on one tile at local {(r, c)}")
            obj[r, c] = OBJECT_INDEX[name]
    return material, obj


# --------------------------------------------------------------------------------------------------------------
# The decoder
def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """uint8 frames ``[B, 64, 64, 3]`` (or already-cropped ``[B, 49, 63, 3]``) -> float ``[B, 3, 49, 63]``."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        raise TypeError("frames must be uint8")
    if frames.shape[-3:-1] == (FRAME_SIZE, FRAME_SIZE):
        frames = world_view_crop(frames)
    if frames.shape[-3:] != (VIEW_ROWS_PX, VIEW_COLS_PX, 3):
        raise ValueError(f"unexpected frame shape {frames.shape}")
    t = torch.from_numpy(np.ascontiguousarray(frames))
    return t.permute(0, 3, 1, 2).float()


class _Block(nn.Module):
    """Pre-activation residual block of two 3 x 3 convolutions on the tile grid."""

    def __init__(self, width: int):
        super().__init__()
        self.c1 = nn.Conv2d(width, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.act(x))))


class TileDecoder(nn.Module):
    """Frame -> per-tile logits for the 7 x 9 world view.

    1. **Tile embedding**: a 7 x 7 convolution with stride 7 aligned to the tile grid (one 147-pixel patch per
       tile -> ``width`` features), plus a learned per-cell position embedding (the night vignette and the player's
       fixed position depend on the cell), then a two-layer per-tile MLP (1 x 1 convolutions).
    2. **Context**: ``blocks`` residual blocks of 3 x 3 convolutions over the tile grid (receptive field
       ``1 + 4 * blocks`` tiles), so a tile drowned in night noise or covered by a sprite is inferred from its
       neighbours.
    3. **Heads**: 1 x 1 convolutions to ``len(MATERIALS)`` material logits and ``len(OBJECTS)`` object logits.

    Input: float ``[B, 3, 49, 63]`` with values 0..255 (``frames_to_tensor``).  Output: ``(material_logits
    [B, 13, 7, 9], object_logits [B, 8, 7, 9])``, indexed like the organ's maps (row 0 north, col 0 west).
    """

    def __init__(self, width: int = 64, blocks: int = 3, mlp_width: int | None = None):
        super().__init__()
        mlp_width = mlp_width or 2 * width
        self.config = {"width": int(width), "blocks": int(blocks), "mlp_width": int(mlp_width)}
        self.embed = nn.Conv2d(3, width, kernel_size=UNIT, stride=UNIT)
        self.pos = nn.Parameter(torch.zeros(1, width, GRID_ROWS, GRID_COLS))
        self.mlp = nn.Sequential(nn.GELU(), nn.Conv2d(width, mlp_width, 1), nn.GELU(), nn.Conv2d(mlp_width, width, 1))
        self.blocks = nn.Sequential(*[_Block(width) for _ in range(blocks)])
        self.out_act = nn.GELU()
        self.material_head = nn.Conv2d(width, len(MATERIALS), 1)
        self.object_head = nn.Conv2d(width, len(OBJECTS), 1)

    def forward(self, x: torch.Tensor):
        x = x * (2.0 / 255.0) - 1.0
        h = self.embed(x) + self.pos
        h = h + self.mlp(h)
        h = self.out_act(self.blocks(h))
        return self.material_head(h), self.object_head(h)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


# --------------------------------------------------------------------------------------------------------------
# The organ
class WeightsMismatch(RuntimeError):
    """The weights file is not the one the model card describes."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def label_space() -> dict:
    return {"material": list(MATERIALS), "object": list(OBJECTS)}


class VisionOrgan:
    """Pinned tile decoder: ``perceive(frame) -> {"material", "object", "material_conf", "object_conf"}``.

    ``material`` / ``object`` are int8 ``[7, 9]`` class indices into ``MATERIALS`` / ``OBJECTS``; the ``*_conf``
    arrays are float32 ``[7, 9]`` softmax probabilities of the predicted class.  Row 0 is north, col 0 is west,
    the player's cell is ``[3, 4]`` (see the module docstring).  The weights are read once, hashed, compared
    with the card's ``weights.sha256`` and only then deserialised from the same bytes; any mismatch -- hash,
    label space or geometry -- raises ``WeightsMismatch``.
    """

    def __init__(self, weights_path, card_path, threads: int | None = 1):
        self.card = json.loads(Path(card_path).read_text())
        data = Path(weights_path).read_bytes()
        digest = sha256_bytes(data)
        expected = self.card.get("weights", {}).get("sha256")
        if digest != expected:
            raise WeightsMismatch(f"weights sha256 {digest} != model card {expected}")
        if self.card.get("label_space") != label_space():
            raise WeightsMismatch("model card label space differs from this organ's")
        geo = self.card.get("geometry", {})
        if {k: geo.get(k) for k in GEOMETRY} != GEOMETRY:
            raise WeightsMismatch("model card geometry differs from this organ's")
        if threads:
            torch.set_num_threads(int(threads))
        self.weights_sha256 = digest
        self.decoder = TileDecoder(**self.card["architecture"]["config"])
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        self.decoder.load_state_dict(state)
        self.decoder.eval()

    @torch.inference_mode()
    def perceive_batch(self, frames: np.ndarray) -> dict:
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[1:] != (FRAME_SIZE, FRAME_SIZE, 3) or frames.dtype != np.uint8:
            raise ValueError(f"expected uint8 frames [B, {FRAME_SIZE}, {FRAME_SIZE}, 3], got "
                             f"{frames.dtype} {frames.shape}")
        ml, ol = self.decoder(frames_to_tensor(frames))
        mp, mi = torch.softmax(ml, 1).max(1)
        op, oi = torch.softmax(ol, 1).max(1)
        return {"material": mi.numpy().astype(np.int8), "object": oi.numpy().astype(np.int8),
                "material_conf": mp.numpy().astype(np.float32), "object_conf": op.numpy().astype(np.float32)}

    def perceive(self, frame: np.ndarray) -> dict:
        frame = np.asarray(frame)
        if frame.shape != (FRAME_SIZE, FRAME_SIZE, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected a uint8 frame [{FRAME_SIZE}, {FRAME_SIZE}, 3], got {frame.dtype} "
                             f"{frame.shape}")
        out = self.perceive_batch(frame[None])
        return {k: v[0] for k, v in out.items()}

    __call__ = perceive


def save_weights(decoder: TileDecoder, path) -> str:
    """Serialise ``decoder``'s state dict to ``path``; return the file's SHA-256."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().contiguous() for k, v in decoder.state_dict().items()}, buf)
    data = buf.getvalue()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return sha256_bytes(data)
