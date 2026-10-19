"""Vision organ: pixel-to-tile geometry against Crafter's renderer, teacher labels against W1's privileged ground
truth (edges, the player tile, night frames, ripe plants), deterministic dataset generation, the weights-hash
gate, and the organ's output contract.  Needs no trained weights (tiny decoders are built and trained here)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from crafter import constants, objects

from continuous_brain_organs import vision as V
from continuous_brain_organs import vision_data as D
from continuous_brain_world import W1Config, W1Env
from continuous_brain_world.baselines import ScriptedAgent
from continuous_brain_world.perception import perceivable_grid

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "continuous_brain_organs" / "cards" / "tile_decoder_v1.json"
A = {name: i for i, name in enumerate(constants.actions)}


def make_env(seed, length=400):
    env = W1Env(W1Config(episode_length=length, arrival_rate0=0.0, arrival_rate_max=0.0, change_rate=0.0))
    obs = env.reset(seed)
    return env, obs


def composite(textures, material, obj):
    """Expected 7 x 7 tile pixels in daylight: Crafter's _draw then _draw_alpha, recomputed here."""
    if material is None:
        base = np.full((V.UNIT, V.UNIT, 3), 127, np.uint8)
    else:
        base = textures.get(material, (V.UNIT, V.UNIT))[..., :3].copy()
    if obj is not None:
        t = textures.get(obj.texture, (V.UNIT, V.UNIT))
        alpha = t[..., 3:].astype(np.float32) / 255
        blended = alpha * (t[..., :3].astype(np.float32) / 255) + (1 - alpha) * (base.astype(np.float32) / 255)
        base = (255 * blended).astype(np.uint8)
    return base.transpose(1, 0, 2)          # textures are x-major like Crafter's canvas; frames are row-major


def daylight_frame(env):
    """The frame Crafter draws for the current state at full daylight (no noise, no tint, RNG untouched)."""
    world = env.bridge.world
    light = world.daylight
    world.daylight = 1.0
    try:
        return env.bridge.render(V.FRAME_SIZE)
    finally:
        world.daylight = light


# --- geometry ------------------------------------------------------------------------------------------------
def test_geometry_is_crafters():
    g = V.crafter_geometry()
    assert (g["unit"], g["cols"], g["rows"], g["item_rows"]) == (7, 9, 7, 2)
    assert (g["border_row"], g["border_col"], g["view_px_rows"], g["view_px_cols"]) == (0, 0, 49, 63)
    assert (V.PLAYER_ROW, V.PLAYER_COL) == (3, 4)
    env, _ = make_env(3)
    assert env.bridge.unit == V.UNIT and env.bridge.view_grid == (V.GRID_COLS, V.GRID_ROWS)
    assert tuple(env.bridge.env._local_view._offset) == (V.PLAYER_COL, V.PLAYER_ROW)
    # other frame sizes follow the same formula (e.g. 128 px -> 14 px tiles, 1 px border)
    g2 = V.crafter_geometry(128)
    assert (g2["unit"], g2["border_row"], g2["border_col"]) == (14, 1, 1)
    assert V.crafter_geometry(100)["border_row"] == (100 - 11 * 9) // 2


def _check_frame_tiles(env, frame):
    world = env.bridge.world
    xs, ys = V.local_to_world(env.bridge.player.pos)
    textures = env.bridge.env._textures
    n_outside = 0
    for r in range(V.GRID_ROWS):
        for c in range(V.GRID_COLS):
            material, obj = world[(xs[r, c], ys[r, c])]
            inside = 0 <= xs[r, c] < world.area[0] and 0 <= ys[r, c] < world.area[1]
            n_outside += not inside
            rs, cs = V.tile_slice(r, c)
            np.testing.assert_array_equal(frame[rs, cs], composite(textures, material if inside else None, obj),
                                          err_msg=f"tile {(r, c)}")
    assert not frame[63].any() and not frame[:, 63].any()          # Crafter's unused last row and column
    return n_outside


def test_pixel_to_tile_geometry_matches_the_renderer_on_every_tile():
    for seed in (1, 2, 5):
        env, _ = make_env(seed)
        rng = np.random.default_rng(seed)
        for _ in range(4):
            _check_frame_tiles(env, daylight_frame(env))
            for _ in range(15):
                env.step(int(rng.integers(1, 5)))
    # at the world's edge the cells beyond it are Crafter's flat grey and the label is 'outside'
    env, _ = make_env(4)
    world, player = env.bridge.world, env.bridge.player
    free = [(x, y) for x in range(0, 3) for y in range(0, 64)
            if world[(x, y)][0] in constants.walkable and world[(x, y)][1] is None]
    assert free, "no free tile near the west edge"
    world.move(player, free[0])
    assert _check_frame_tiles(env, daylight_frame(env)) > 0
    mat, _ = V.labels_from_privileged(env.privileged_now()["privileged"])
    assert (mat[:, 0] == V.MATERIAL_INDEX["outside"]).all()


def test_world_view_crop_and_tensor_layout():
    frame = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    crop = V.world_view_crop(frame)
    assert crop.shape == (49, 63, 3) and np.shares_memory(crop, frame)
    np.testing.assert_array_equal(crop, frame[:49, :63])
    t = V.frames_to_tensor(frame[None])
    assert t.shape == (1, 3, 49, 63) and t.dtype == torch.float32
    np.testing.assert_array_equal(t[0].permute(1, 2, 0).numpy(), frame[:49, :63].astype(np.float32))
    with pytest.raises(TypeError):
        V.frames_to_tensor(frame[None].astype(np.float32))


# --- teacher labels --------------------------------------------------------------------------------------------
def _independent_labels(env, priv):
    """Ground truth recomputed another way: W1's perception grid for positions, Crafter's semantic view for ids,
    and each object's own texture name for the object class."""
    pos, _ = perceivable_grid(env.bridge)               # [x, y, 2], W1's own window geometry
    pos = pos.transpose(1, 0, 2)                          # -> [row, col, 2]
    sem = priv["semantic"]
    world = env.bridge.world
    n_mat = len(world._mat_ids)
    sem_cls = [objects.Player, objects.Cow, objects.Zombie, objects.Skeleton, objects.Arrow, objects.Plant]
    mat = np.zeros((V.GRID_ROWS, V.GRID_COLS), np.int8)
    obj = np.zeros_like(mat)
    for r in range(V.GRID_ROWS):
        for c in range(V.GRID_COLS):
            x, y = (int(v) for v in pos[r, c])
            if not (0 <= x < 64 and 0 <= y < 64):
                continue
            mat[r, c] = priv["material_map"][x, y]
            o = world[(x, y)][1]
            if o is None:
                assert sem[x, y] == mat[r, c]
                continue
            assert sem[x, y] == n_mat + sem_cls.index(type(o))
            tex = o.texture
            name = {"cow": "cow", "zombie": "zombie", "skeleton": "skeleton", "plant": "sapling",
                    "plant-ripe": "plant_ripe"}.get(tex)
            name = name or ("player" if tex.startswith("player") else "arrow" if tex.startswith("arrow") else None)
            obj[r, c] = V.OBJECT_INDEX[name]
    return mat, obj


def test_labels_match_privileged_ground_truth_day_night_edges_player():
    seen_night = seen_edge_obj = 0
    kinds = set()
    for seed in (11, 12):
        env = W1Env(W1Config(episode_length=400))
        obs = env.reset(seed)
        agent = ScriptedAgent(seed=seed)
        info = env.privileged_now()
        for t in range(300):
            if t % 3 == 0 or 148 <= obs["step"] <= 160:
                priv = info["privileged"]
                mat, obj = V.labels_from_privileged(priv)
                emat, eobj = _independent_labels(env, priv)
                np.testing.assert_array_equal(mat, emat)
                np.testing.assert_array_equal(obj, eobj)
                assert obj[V.PLAYER_ROW, V.PLAYER_COL] == V.OBJECT_INDEX["player"]
                assert (obj == V.OBJECT_INDEX["player"]).sum() == 1
                seen_night += priv["daylight"] < 0.5
                edge = np.concatenate([obj[0], obj[-1], obj[:, 0], obj[:, -1]])
                seen_edge_obj += int((edge > 0).sum())
                kinds |= set(np.unique(obj).tolist()) | {100 + m for m in np.unique(mat).tolist()}
            a, u = agent.act(obs, info)
            obs, _, done, info = env.step(a, u)
            if done:
                break
    assert seen_night > 0 and seen_edge_obj > 0
    assert len(kinds) >= 8                                  # material and object classes seen in those frames


def test_ripe_plant_and_sapling_labels():
    env, _ = make_env(6)
    world, player = env.bridge.world, env.bridge.player
    target = None
    for dx in (-2, 2, -1, 1):
        p = (int(player.pos[0]) + dx, int(player.pos[1]) + 1)
        if world[p][0] == "grass" and world[p][1] is None:
            target = p
            break
    assert target is not None
    plant = objects.Plant(world, target)
    world.add(plant)
    r, c = target[1] - int(player.pos[1]) + V.PLAYER_ROW, target[0] - int(player.pos[0]) + V.PLAYER_COL
    _, obj = V.labels_from_privileged(env.privileged_now()["privileged"])
    assert obj[r, c] == V.OBJECT_INDEX["sapling"] and plant.texture == "plant"
    plant.grown = 301
    _, obj = V.labels_from_privileged(env.privileged_now()["privileged"])
    assert obj[r, c] == V.OBJECT_INDEX["plant_ripe"] and plant.texture == "plant-ripe"
    _check_frame_tiles(env, daylight_frame(env))         # the renderer draws exactly that texture there


def test_label_convention_row0_is_north_col0_is_west():
    env, obs = make_env(8)
    priv = env.privileged_now()["privileged"]
    mat0, _ = V.labels_from_privileged(priv)
    px, py = priv["player_pos"]
    mm = priv["material_map"]
    assert mat0[0, V.PLAYER_COL] == mm[px, py - 3] and mat0[6, V.PLAYER_COL] == mm[px, py + 3]
    assert mat0[V.PLAYER_ROW, 0] == mm[px - 4, py] and mat0[V.PLAYER_ROW, 8] == mm[px + 4, py]
    # walk north (move_up) until the player actually moves: the map shifts one row south
    for _ in range(20):
        obs, _, _, info = env.step(A["move_up"])
        p2 = info["privileged"]["player_pos"]
        if p2 != (px, py):
            break
    assert p2 == (px, py - 1)
    mat1, _ = V.labels_from_privileged(info["privileged"])
    np.testing.assert_array_equal(mat1[1:], mat0[:-1])


def test_unknown_object_kind_is_refused():
    env, _ = make_env(9)
    priv = env.privileged_now()["privileged"]
    priv["objects"].append({"kind": "fence", "pos": priv["player_pos"], "health": 0, "ripe": False})
    with pytest.raises(ValueError):
        V.labels_from_privileged(priv)


# --- dataset generation -----------------------------------------------------------------------------------------
def test_dataset_generation_is_deterministic():
    for behaviour in ("explorer", "gardener", "scripted", "random"):
        a = D.generate_episode(5150, behaviour, 70, 2)
        b = D.generate_episode(5150, behaviour, 70, 2)
        assert D.digest(a) == D.digest(b), behaviour
        assert len(a["frames"]) > 5 and a["frames"].shape[1:] == (64, 64, 3)
        assert (a["behaviour"] == D.BEHAVIOURS.index(behaviour)).all()
    c = D.generate_episode(5151, "explorer", 70, 2)
    assert D.digest(c) != D.digest(D.generate_episode(5150, "explorer", 70, 2))


def test_dataset_frames_are_the_observations_and_labels_are_the_teachers():
    d = D.generate_episode(5152, "explorer", 60, 3)
    env = W1Env(W1Config(episode_length=60, privileged_semantic=False))
    obs = env.reset(5152)
    agent = D.ExplorerAgent(seed=5152)
    agent.reset(obs)
    agent.assist(env, first=True)
    info = env.privileged_now()
    k = 0
    while True:
        if obs["step"] % 3 == 0:
            np.testing.assert_array_equal(d["frames"][k], obs["image"])
            mat, obj = V.labels_from_privileged(info["privileged"])
            np.testing.assert_array_equal(d["material"][k], mat)
            np.testing.assert_array_equal(d["object"][k], obj)
            assert d["step"][k] == obs["step"]
            k += 1
        if env.done:
            break
        a, u = agent.act(obs, info)
        obs, _, done, info = env.step(a, u)
        if not done:
            agent.assist(env)
    assert k == len(d["frames"])


def test_explorer_survives_and_never_stands_on_lava():
    d = D.generate_episode(5153, "explorer", 200, 1)
    assert len(d["frames"]) == 201                        # alive to the step limit
    assert (d["material"][:, V.PLAYER_ROW, V.PLAYER_COL] != V.MATERIAL_INDEX["lava"]).all()


def test_split_round_trip(tmp_path):
    d = D.generate_episode(5154, "random", 30, 1)
    man = D.save_split(d, tmp_path / "s", {"x": 1})
    back = D.load_split(tmp_path / "s")
    assert man["sha256"] == D.digest(back) == D.digest(d)


# --- the organ -------------------------------------------------------------------------------------------------
def tiny_card(tmp_path, decoder):
    sha = V.save_weights(decoder, tmp_path / "w.pt")
    card = {"name": "tiny", "architecture": {"config": decoder.config}, "label_space": V.label_space(),
            "geometry": dict(V.GEOMETRY), "weights": {"sha256": sha}}
    (tmp_path / "card.json").write_text(json.dumps(card))
    return tmp_path / "w.pt", tmp_path / "card.json"


def test_organ_refuses_weights_whose_hash_differs(tmp_path):
    torch.manual_seed(0)
    w, c = tiny_card(tmp_path, V.TileDecoder(width=8, blocks=1))
    organ = V.VisionOrgan(w, c)
    assert organ.weights_sha256 == json.loads(c.read_text())["weights"]["sha256"]
    data = bytearray(w.read_bytes())
    data[len(data) // 2] ^= 1
    bad = tmp_path / "bad.pt"
    bad.write_bytes(bytes(data))
    with pytest.raises(V.WeightsMismatch):
        V.VisionOrgan(bad, c)
    other = tmp_path / "other"
    other.mkdir()
    w2, _ = tiny_card(other, V.TileDecoder(width=8, blocks=1))   # valid weights, but not the card's
    with pytest.raises(V.WeightsMismatch):
        V.VisionOrgan(w2, c)
    card = json.loads(c.read_text())
    card["label_space"]["object"] = card["label_space"]["object"][:-1]
    c.write_text(json.dumps(card))
    with pytest.raises(V.WeightsMismatch):
        V.VisionOrgan(w, c)


def test_organ_output_contract(tmp_path):
    torch.manual_seed(0)
    w, c = tiny_card(tmp_path, V.TileDecoder(width=8, blocks=1))
    organ = V.VisionOrgan(w, c)
    env, obs = make_env(10)
    out = organ.perceive(obs["image"])
    assert set(out) == {"material", "object", "material_conf", "object_conf"}
    for k, dt in (("material", np.int8), ("object", np.int8), ("material_conf", np.float32),
                  ("object_conf", np.float32)):
        assert out[k].shape == (V.GRID_ROWS, V.GRID_COLS) and out[k].dtype == dt, k
    assert out["material"].min() >= 0 and out["material"].max() < len(V.MATERIALS)
    assert out["object"].min() >= 0 and out["object"].max() < len(V.OBJECTS)
    for k in ("material_conf", "object_conf"):
        assert (out[k] > 0).all() and (out[k] <= 1).all()
        assert (out[k] >= 1 / (len(V.MATERIALS) if k == "material_conf" else len(V.OBJECTS)) - 1e-6).all()
    batch = organ.perceive_batch(np.stack([obs["image"]] * 3))
    assert batch["material"].shape == (3, V.GRID_ROWS, V.GRID_COLS)
    np.testing.assert_array_equal(batch["material"][1], out["material"])
    with pytest.raises(ValueError):
        organ.perceive(obs["image"][:49])
    with pytest.raises(ValueError):
        organ.perceive(obs["image"].astype(np.float32))


def test_a_few_training_steps_learn_the_player_cell_in_the_organs_convention(tmp_path):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    d = D.generate_episode(5155, "explorer", 240, 1)
    dec = V.TileDecoder(width=16, blocks=1)
    opt = torch.optim.Adam(dec.parameters(), 5e-3)
    x = V.frames_to_tensor(d["frames"])
    yo = torch.from_numpy(d["object"].astype(np.int64))
    ym = torch.from_numpy(d["material"].astype(np.int64))
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        idx = torch.randint(0, len(x), (32,), generator=g)
        ml, ol = dec(x[idx])
        loss = torch.nn.functional.cross_entropy(ml, ym[idx]) + torch.nn.functional.cross_entropy(ol, yo[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    w, c = tiny_card(tmp_path, dec)
    organ = V.VisionOrgan(w, c)
    out = organ.perceive_batch(d["frames"][::10])
    assert (out["object"][:, V.PLAYER_ROW, V.PLAYER_COL] == V.OBJECT_INDEX["player"]).mean() > 0.9
    others = np.ones((V.GRID_ROWS, V.GRID_COLS), bool)
    others[V.PLAYER_ROW, V.PLAYER_COL] = False
    assert (out["object"][:, others] == V.OBJECT_INDEX["player"]).mean() < 0.02
    assert (out["material"] == d["material"][::10]).mean() > 0.9     # maps come back in the labels' [row, col]


def test_committed_model_card_is_consistent_with_the_code():
    if not CARD.exists():
        pytest.skip("model card not generated yet")
    card = json.loads(CARD.read_text())
    assert card["label_space"] == V.label_space()
    assert {k: card["geometry"][k] for k in V.GEOMETRY} == V.GEOMETRY
    dec = V.TileDecoder(**card["architecture"]["config"])
    assert dec.n_parameters() == card["architecture"]["n_parameters"]
    assert len(card["weights"]["sha256"]) == 64
    tr = card["training"]["splits"]
    seeds = {s: set(range(tr[s]["seeds"][0], tr[s]["seeds"][1] + 1)) for s in ("train", "val", "test")}
    assert not (seeds["train"] & seeds["val"]) and not (seeds["train"] & seeds["test"]) \
        and not (seeds["val"] & seeds["test"])
    weights = REPO / card["weights"]["file"]
    if weights.exists():                                   # only on the machine that trained it
        organ = V.VisionOrgan(weights, CARD)
        assert organ.weights_sha256 == card["weights"]["sha256"]


def test_training_is_deterministic():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_vision_organ",
                                                  REPO / "scripts" / "explore" / "train_vision_organ.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = D.generate_episode(5156, "explorer", 90, 1)
    cfg = dict(mod.TRAIN, width=8, blocks=1, epochs=1, batch=16, warmup_steps=2)
    try:
        a, ia = mod.train(data, data, cfg)
        b, ib = mod.train(data, data, cfg)
    finally:
        torch.use_deterministic_algorithms(False)
    assert ia["steps"] == ib["steps"] == len(data["frames"]) // 16
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka
    assert ia["history"][0]["val"] == ib["history"][0]["val"]
