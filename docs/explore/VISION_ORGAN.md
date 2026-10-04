# The vision organ (exploration lane)

**Status:** engineering result on the exploration lane (`explore/w1-vision`), not preregistered.
Code: `continuous_brain_organs/vision.py` (organ, decoder, geometry, teacher labels),
`continuous_brain_organs/vision_data.py` (developmental dataset). Training and evaluation:
`scripts/explore/train_vision_organ.py`. Model card: `continuous_brain_organs/cards/tile_decoder_v1.json`.
Evaluation report: `evidence/explore/vision/tile_decoder_v1_eval.json`. Tests: `tests/test_vision_organ.py`.

The persistent individual perceives W1 from **pixels only**. Its body sense (inventory, vitals, position,
facing, sleeping) arrives separately. The vision organ turns each 64 x 64 x 3 frame into a **perceived local tile
map**: for every tile of the area Crafter shows, which material is there and which object (if any) stands on it,
each with a confidence. It never reads the privileged material map or object list at run time.

## 1. What the frame shows: the geometry (verified from Crafter's source and pixel-for-pixel)

Crafter 1.8.3 renders with `Env(area=(64, 64), view=(9, 9), size=(64, 64))` (W1's settings):

| quantity | formula in crafter | value |
|---|---|---|
| pixels per tile | `unit = size // view` | 64 // 9 = **7** |
| inventory rows | `ceil(len(items) / view[0])` | ceil(16 / 9) = **2** |
| world view grid | `LocalView(grid=[view[0], view[1] - item_rows])` | **9 columns x 7 rows** |
| inventory strip | `ItemView(grid=[9, 2])` | 9 x 2 tiles below the view |
| border | `(size - unit * view) // 2` | **0** (63 x 63 drawn at the top-left; row 63 and column 63 stay black) |
| axis order | the canvas is x-major and `render` returns `canvas.transpose(1, 0, 2)` | pixel rows are world **y** |
| centring | `pos = player.pos + [i, j] - grid // 2` | player at grid **(col 4, row 3)** |

So the world view is `frame[0:49, 0:63]`; tile (row `i`, col `j`) covers `frame[7i:7i+7, 7j:7j+7]` and shows the
world tile `(x, y) = (player_x + j - 4, player_y + i - 3)`. The view is 9 wide and 7 tall: 4 tiles to the west
and east, 3 to the north and south. Cells beyond the world's edge are not drawn (flat grey 127).

`crafter_geometry()` recomputes all of this from Crafter's own formulas; the tests compare every tile of real
frames (several seeds, after walks, and at the world's west edge) byte-for-byte with Crafter's textures
composited exactly as `_draw` / `_draw_alpha` do, at full daylight.

**Conventions of the organ's maps** (arrays `[7, 9]`, indexed `[row, col]`):

- row 0 is **north** (smallest world y; `move_up` is `(0, -1)`), row 6 south;
- col 0 is **west** (smallest world x; `move_left` is `(-1, 0)`), col 8 east;
- the player is at **`[3, 4]`**; `local_to_world(player_pos)` gives each cell's absolute `(x, y)`.

W1's own memory questions use offsets with `+y` = south; a cell `[r, c]` is `(c - 4) east, (r - 3) south` of the
player.

## 2. What it perceives: the label spaces

- **material** (13; the index *is* Crafter's material id): `outside` (id 0, Crafter's `None`: beyond the world
  edge -- geometry, not darkness), water, grass, stone, path, sand, tree, lava, coal, iron, diamond, table,
  furnace.
- **object** (8): `none`, `player`, `cow`, `zombie`, `skeleton`, `arrow` (one class for the four directions),
  `sapling` (`objects.Plant` with `grown <= 300`, texture `plant`), `plant_ripe` (`grown > 300`, texture
  `plant-ripe`). Crafter defines `Fence` too, but nothing in 1.8.3 can create one (no worldgen rule, no `place`
  recipe), so it is not a class and the label extractor refuses unknown objects.

Objects are sprites alpha-blended over their tile's material, so both heads are defined on every tile,
including the player's. Darkness is **not** a class: at night the organ must still say what is there; the
evaluation reports night separately (and splits night tiles by whether W1's visibility rule counts them as
perceivable).

## 3. The organ

`VisionOrgan(weights_path, card_path)` reads the weights file once, hashes it, compares the SHA-256 with the
card's `weights.sha256`, checks the card's label space and geometry against the code, and only then
deserialises the state dict from the same bytes (`torch.load(weights_only=True)`). Any mismatch raises
`WeightsMismatch` (tested with a flipped byte, with valid weights that are not the card's, and with a card
whose label space differs).

`organ.perceive(frame)` takes a uint8 `[64, 64, 3]` frame and returns

```
{"material": int8[7, 9], "object": int8[7, 9], "material_conf": float32[7, 9], "object_conf": float32[7, 9]}
```

(class indices into `MATERIALS` / `OBJECTS`; confidences are the softmax probability of the predicted class).
`perceive_batch` does the same for `[B, 64, 64, 3]`.

**`TileDecoder`** (`tile_decoder_v1`: width 64, 3 blocks, 253,013 parameters): a 7 x 7 convolution with stride 7
aligned to the tile grid embeds each tile's 147 pixels; a learned per-cell position embedding is added (the
night vignette and the player's fixed cell depend on position); a two-layer per-tile MLP (1 x 1 convolutions);
three residual blocks of two 3 x 3 convolutions over the 7 x 9 tile grid (receptive field 13 tiles: the whole
view), so a tile drowned in night noise or under a sprite is inferred from its surroundings; 1 x 1 material and
object heads. Input is the world view only (`frame[0:49, 0:63]`): the inventory strip is body sense, not vision.

## 4. The supervised developmental phase (and what replaces it)

The decoder is trained with a **teacher**: labels are W1's privileged ground truth at the instant of each frame
(`labels_from_privileged(info["privileged"])`: the material map and the objects inside the visible window). This
is a developmental phase -- like a caretaker naming things for a child -- and is documented as such in the model
card. It is a shortcut, not the end state: it presumes a simulator that can tell the truth, and it fixes the
categories to Crafter's.

What replaces it, in order of ambition:

1. **Self-supervised learning on the individual's own experience**: predict the next frame or masked tiles
   from action and context (a world model), learn tile embeddings by contrastive or JEPA-style objectives, and
   name categories later from sparse caretaker language ("that is a cow"). Labels then come from the individual's
   life, not from the simulator.
2. **Pretrained open-weight vision** (SigLIP / CLIP-class encoders, DINOv2) on a GPU machine, with a small
   tile-aligned probe on top -- the right choice for W2's 3D egocentric camera, where hand-made tile labels stop
   existing. Not possible here: no GPU and no Hugging Face access.

The organ's contract (frame in, pinned weights, a map with confidences out) stays the same either way.

## 5. Developmental data

`continuous_brain_organs/vision_data.py`, seed family `w1-vision-eng`, one fresh world per episode, disjoint seed
ranges per split (disjoint also from the W1 baselines' 71001-71020 and the scripted agent's tuning seeds 1-12):

| split | seeds | episodes (scripted / explorer / gardener / random) | frames | tiles | night frames | sleeping frames | generation |
|---|---|---|---|---|---|---|---|
| train | 72001-72142 | 36 / 50 / 16 / 40 | 67,726 | 4,266,738 | 26,270 | 5,813 | 260 s |
| val | 73001-73015 | 4 / 5 / 2 / 4 | 7,708 | 485,604 | 2,981 | 751 | 30 s |
| test | 74001-74030 | 8 / 10 / 4 / 8 | 14,481 | 912,303 | 5,626 | 1,263 | 51 s |

(Generation used 2 worker processes; episodes are capped at 2000 steps (scripted), 1200 (explorer), 1500
(gardener), 600 (random). Training-set tile counts, material: grass 2.29 M, stone 530 k, water 472 k, path 393 k,
sand 209 k, tree 202 k, coal 46 k, outside 40 k, lava 34 k, iron 18 k, table 14 k, furnace 13 k, diamond 4.0 k;
object: cow 18 k, zombie 41 k, skeleton 19 k, arrow 10.7 k, sapling 15.7 k, plant_ripe 1.3 k.)

Behaviours: **scripted** = W1's privileged scripted baseline (tables, furnaces, saplings, mining, fighting,
shelters, sleeping at night); **explorer** = a teacher-assisted exploratory policy (granted pickaxes, swords,
wood, stone and saplings; vitals and health topped up every step so it roams caves, tunnels, ore and lava fields
and stays out at night; never steps on lava; 10 % random actions); **gardener** = the same policy bound to 9 tiles
around its spawn point, planting saplings, so plants stay within Crafter's object-update radius long enough to
ripen; **random** = W1's uniform random baseline. The teacher edits only the player's inventory, never the world;
the inventory strip is not the organ's input. Every `stride`-th observation (3 for scripted, 2 for explorer and
gardener, 1 for random) is kept, with the full 64 x 64 frame, both label maps, daylight, sleeping, position and
W1's per-tile perceivability. The data lives in `runs/explore/vision/data_v1/` (gitignored, about 1.1 GB;
each split's manifest records its SHA-256); regeneration is deterministic (tested).

## 6. Training

Seeded (`torch.manual_seed(1)`, seeded permutation), `torch.use_deterministic_algorithms(True)`, 2 torch
threads, AdamW (lr 2e-3, weight decay 1e-4), 200 warm-up steps then cosine decay to 0, batch 256, **12 epochs =
3,168 steps (811 k samples)**, unweighted cross-entropy on both heads, no augmentation. **Training wall time
560 s (9.3 min)** on this 2-CPU machine (data generation 341 s more, once; the script reuses it with
`--reuse-data`). Two engineering findings that decide whether training fits the budget at all:

- **Subnormal floats.** Without `torch.set_flush_denormal(True)` epochs slowed down progressively
  (85 s, 185 s, > 300 s for the first three) as weights, gradients and Adam moments decayed into the subnormal
  range; with it every epoch takes 46-47 s.
- **Shared CPUs.** Another lane's jobs ran on the same two CPUs at times. With OpenMP's default spin-waiting,
  2 torch threads trained at 142 samples/s while another process held a CPU; with `OMP_WAIT_POLICY=PASSIVE`
  (set by the script) 827 samples/s.

Validation (seeds 73001-73015) after epochs 1 / 3 / 6 / 12: material 0.961 / 0.988 / 0.993 / 0.995, object
0.988 / 0.997 / 0.998 / 0.999, night material 0.914 / 0.971 / 0.984 / 0.989. The final weights are kept (no
selection on validation). The test seeds were used once before, in a 4-epoch pipeline smoke run at 12 % scale
that only checked the pipeline end to end; no design choice used test numbers (the later changes -- epoch count,
subnormal flushing, OpenMP waiting -- were about wall time). Weights: `runs/explore/vision/
tile_decoder_v1.pt` (gitignored, 1.02 MB), SHA-256 `0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999`.

## 7. Evaluation (held-out test seeds 74001-74030)

All numbers are computed by running the real `VisionOrgan` (hash check included) on the full observations of the
test split. Full report with every class, both confusion matrices and confidence bands:
`evidence/explore/vision/tile_decoder_v1_eval.json`.

14,481 frames, 912,303 tiles, 30 worlds never seen in training.

| tiles | n | material | object | both |
|---|---|---|---|---|
| all | 912,303 | **0.9950** | **0.9985** | 0.9936 |
| day (daylight >= 0.5) | 557,865 | 0.9988 | 0.9998 | 0.9986 |
| night (daylight < 0.5) | 354,438 | 0.9889 | 0.9965 | 0.9858 |
| night, perceivable by W1's rule | 217,140 | 0.9982 | 0.9993 | 0.9974 |
| night, drowned in noise | 137,298 | 0.9742 | 0.9922 | 0.9674 |
| player sleeping | 79,569 | 0.9657 | 0.9973 | 0.9634 |
| player tile [3, 4] | 14,481 | 0.9995 | 1.0000 | 0.9995 |
| other tiles | 897,822 | 0.9949 | 0.9985 | 0.9935 |
| view edge (rows 0, 6; cols 0, 8) | 405,468 | 0.9909 | 0.9977 | 0.9889 |
| interior | 506,835 | 0.9982 | 0.9992 | 0.9975 |

Whole frames with all 63 tiles right: material 0.890, object 0.929. Macro F1: material 0.978, object 0.932.
By behaviour (material / object): scripted 0.988 / 0.999, explorer 0.997 / 0.998, gardener 0.998 / 0.998, random
0.999 / 1.000 (scripted is lowest because it spends its nights asleep in shelters).

**Per class** (precision / recall / F1, support in tiles; night recall in the last column):

| material | support | P | R | F1 | night R |
|---|---|---|---|---|---|
| outside | 8,135 | 0.988 | 0.898 | 0.941 | 0.785 |
| water | 94,722 | 0.997 | 0.998 | 0.998 | 0.996 |
| grass | 517,653 | 0.998 | 0.999 | 0.998 | 0.997 |
| stone | 99,926 | 0.989 | 0.995 | 0.992 | 0.988 |
| path | 85,249 | 0.988 | 0.988 | 0.988 | 0.972 |
| sand | 39,540 | 0.988 | 0.996 | 0.992 | 0.990 |
| tree | 41,963 | 0.994 | 0.993 | 0.993 | 0.981 |
| lava | 5,890 | 0.991 | 0.995 | 0.993 | 0.991 |
| coal | 8,826 | 0.992 | 0.976 | 0.984 | 0.945 |
| iron | 3,214 | 0.977 | 0.939 | 0.958 | 0.865 |
| diamond | 685 | 0.975 | 0.952 | 0.963 | 0.933 |
| table | 3,388 | 0.977 | 0.926 | 0.951 | 0.834 |
| furnace | 3,112 | 0.988 | 0.944 | 0.966 | 0.890 |

| object | support | P | R | F1 | night R |
|---|---|---|---|---|---|
| none | 873,502 | 0.999 | 1.000 | 0.999 | 1.000 |
| player | 14,481 | 0.999 | 1.000 | 1.000 | 1.000 |
| cow | 4,074 | 0.977 | 0.956 | 0.967 | 0.868 |
| zombie | 9,091 | 0.990 | 0.967 | 0.979 | 0.954 |
| skeleton | 4,391 | 0.990 | 0.946 | 0.967 | 0.862 |
| arrow | 2,682 | 0.983 | 0.882 | 0.930 | 0.742 |
| sapling | 3,932 | 0.994 | 0.976 | 0.985 | 0.934 |
| plant_ripe | 150 | 0.959 | 0.467 | 0.628 | 0.493 |

**Rare classes -- read with the support.** Tile counts are not independent samples: the same diamond seen in
40 consecutive kept frames is 40 tiles. Diamond: 685 tiles (recall day 0.966 of 386, night 0.933 of 299).
Lava: 5,890 (day 0.999, night 0.991). Iron: 3,214 (day 1.000, night 0.865). Skeleton: 4,391 (day 1.000, night
0.862). Arrow: 2,682 (day 0.970, night 0.742). **Ripe plant: 150 tiles from only 5 test worlds** -- recall
0.467 (day 0.443 of 79, night 0.493 of 71); misses go to cow (30), none (28) and sapling (20). The ripe and young
plant textures differ in only 7 of 49 pixels at 7 px per tile, and the training set holds just 1,254 ripe-plant
tiles; this class is not usable yet (section 8).

**Confidence is informative**: tiles with confidence >= 0.99 (889,878 material / 897,376 object) are 0.9996 /
0.9997 correct; 0.9-0.99: 0.925 / 0.967; 0.5-0.9: 0.702 / 0.753; below 0.5: 0.418 / 0.443.

**Inference speed** (the real `VisionOrgan.perceive`, end to end, one torch thread): **781 frames/s** for single
frames, 2,935 frames/s in batches of 256 (2,000 test frames).

## 8. Failure modes

- **Ripe plants** (recall 0.47, 150 test tiles from 5 worlds): too few training examples (ripening needs a
  plant to stay within 18 tiles of the player for 300 steps without a cow or monster beside it) and a 7-pixel
  difference from a young plant. Next step: oversample frames that contain ripe plants (or weight the loss), and
  generate more gardener episodes; an organ consumer should treat `plant_ripe` vs `sapling` as unreliable until
  then.
- **Night.** Accuracy falls from 0.999 (day) to 0.989 (material). Where W1's visibility rule says the true image
  still dominates the noise, night is as good as day (0.998); the loss is in the drowned corners (0.974), where
  the decoder is guessing from context: `outside` (night recall 0.78), tables (0.83), furnaces (0.89), iron
  (0.86), cows (0.87), skeletons (0.86) and arrows (0.74). Those predictions carry lower confidence -- use it.
- **Sleeping** (0.966 material): the frame is desaturated and blue-tinted and the scripted agent sleeps at night
  in stone pockets, so sleep stacks on darkness.
- **Occlusion and sprites.** Objects are alpha-blended over their tile, so a sprite hides part of the material;
  the player tile is still 0.9995 (the player stands on grass, sand or path). A cow, zombie or arrow on an unusual
  material is rarer and less certain.
- **Small objects at the edge**: edge tiles are 0.991 (material) vs 0.998 inside; the 3 x 3 context of an edge
  tile is cut off, and at night the edges are the darkest.
- **Distribution.** Trained on four behaviour policies in fresh 64 x 64 Crafter worlds at 64 x 64 pixels. A new
  texture pack, a different frame size (`crafter_geometry` handles the geometry, the weights do not) or a long
  life that reshapes the world (large mined-out areas, many placed tables) are out of distribution and need
  re-evaluation. Arrows are one class (direction is not perceived).

## 9. How the individual uses it

The organ is a **host**: a fixed function with pinned weights, run once per tick on the observation the
individual received. Its output map (both label maps and both confidence maps, plus the card's name and weights
SHA-256 so the provenance of every percept is recorded) is written to the canonical store as that tick's
percept, alongside the body sense. Downstream organs (memory, the obligation tracker, the controller) read the
percept from the store, never the frame or the privileged state; memory answers ("where did you last see a
cow?") are built from accumulated percepts translated to world coordinates with `local_to_world` and the
body sense's position. Low-confidence cells (night, sprites) should be stored as such and not overwrite a
confident earlier percept of a static tile; that fusion belongs to the memory organ, not to vision.
A new set of weights is a new organ version: a new card, a new hash, recorded in the store from the tick it is
installed.

## 10. Reproduce

```
/home/claude/venvs/explore/bin/python -m pytest -q -p no:cacheprovider tests/test_vision_organ.py
/home/claude/venvs/explore/bin/python scripts/explore/train_vision_organ.py               # data + training + eval
/home/claude/venvs/explore/bin/python scripts/explore/train_vision_organ.py --reuse-data  # skip regeneration
```
