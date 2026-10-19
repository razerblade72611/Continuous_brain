#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
)


GRID = 0.25


@dataclass(frozen=True)
class DoorEpisode:
    slot: int
    label: str
    object_id: str
    position: tuple[float, float, float]
    outward_axis: str
    outward_sign: int
    approach: tuple[float, float, float]
    far_target: tuple[float, float, float]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_controller(executable, scene, width=160, height=120):
    from ai2thor.controller import Controller

    common = {
        "scene": str(scene),
        "width": int(width),
        "height": int(height),
        "renderDepthImage": True,
    }

    last = None
    for extra in (
        {"local_executable_path": str(executable)},
        {"executable_path": str(executable)},
    ):
        try:
            return Controller(**common, **extra)
        except TypeError as exc:
            last = exc

    raise last


def metadata(controller):
    return dict(getattr(controller.last_event, "metadata", {}) or {})


def objects(controller):
    return list(metadata(controller).get("objects", []) or [])


def object_by_id(controller, object_id):
    for obj in objects(controller):
        if str(obj.get("objectId")) == str(object_id):
            return obj
    return None


def action_ok(event, label):
    md = dict(getattr(event, "metadata", {}) or {})
    if not bool(md.get("lastActionSuccess", True)):
        raise RuntimeError(
            f"{label} failed: {md.get('errorMessage', '')}"
        )
    return event


def agent_pose(controller):
    agent = metadata(controller).get("agent", {}) or {}
    pos = agent.get("position", {}) or {}
    rot = agent.get("rotation", {}) or {}
    return {
        "x": float(pos.get("x", 0.0)),
        "y": float(pos.get("y", 0.0)),
        "z": float(pos.get("z", 0.0)),
        "yaw": float(rot.get("y", 0.0)) % 360.0,
        "horizon": float(agent.get("cameraHorizon", 0.0)),
    }


def door_rows(controller):
    rows = []
    for obj in objects(controller):
        if str(obj.get("objectType", "")).lower() != "door":
            continue
        p = obj.get("position", {}) or {}
        rows.append(
            {
                "objectId": str(obj.get("objectId")),
                "openable": bool(obj.get("openable", False)),
                "isOpen": bool(obj.get("isOpen", False)),
                "position": {
                    "x": float(p.get("x", 0.0)),
                    "y": float(p.get("y", 0.0)),
                    "z": float(p.get("z", 0.0)),
                },
            }
        )
    return rows


def reachable(controller):
    ev = action_ok(
        controller.step(action="GetReachablePositions"),
        "GetReachablePositions",
    )
    pts = list(ev.metadata.get("actionReturn", []) or [])
    if not pts:
        raise RuntimeError("no reachable positions")
    return [
        {
            "x": float(p.get("x", 0.0)),
            "y": float(p.get("y", 0.0)),
            "z": float(p.get("z", 0.0)),
        }
        for p in pts
    ]


def ensure_door_state(controller, object_id, want_open, *, force=True):
    obj = object_by_id(controller, object_id)
    if obj is None:
        raise RuntimeError(f"missing door {object_id}")

    if bool(obj.get("isOpen", False)) == bool(want_open):
        return True

    action = "OpenObject" if want_open else "CloseObject"
    kwargs = {
        "action": action,
        "objectId": str(object_id),
    }
    if force:
        kwargs["forceAction"] = True

    ev = controller.step(**kwargs)
    obj = object_by_id(controller, object_id)
    return bool(
        obj is not None
        and bool(obj.get("isOpen", False)) == bool(want_open)
    )


def native_open_door(controller, episode: DoorEpisode):
    face_point(
        controller,
        episode.position[0],
        episode.position[2],
    )
    ev = controller.step(
        action="OpenObject",
        objectId=episode.object_id,
    )
    obj = object_by_id(controller, episode.object_id)
    return {
        "action_success": bool(
            (getattr(ev, "metadata", {}) or {})
            .get("lastActionSuccess", False)
        ),
        "is_open": bool(obj and obj.get("isOpen", False)),
        "error": str(
            (getattr(ev, "metadata", {}) or {})
            .get("errorMessage", "")
        ),
    }


def qkey(point, grid=GRID):
    return (
        int(round(float(point["x"]) / float(grid))),
        int(round(float(point["z"]) / float(grid))),
    )


def point_tuple(point):
    return (
        float(point["x"]),
        float(point.get("y", 0.0)),
        float(point["z"]),
    )


def index_reachable(points, grid=GRID):
    return {qkey(p, grid): dict(p) for p in points}


def nearest_key(index, x, z):
    if not index:
        raise RuntimeError("empty reachability index")

    target = (float(x), float(z))
    return min(
        index,
        key=lambda k: (
            (float(index[k]["x"]) - target[0]) ** 2
            + (float(index[k]["z"]) - target[1]) ** 2
        ),
    )


def bfs_path(points, start_xyz, goal_xyz, grid=GRID):
    index = index_reachable(points, grid)
    start = nearest_key(
        index, float(start_xyz[0]), float(start_xyz[2])
    )
    goal = nearest_key(
        index, float(goal_xyz[0]), float(goal_xyz[2])
    )

    if start == goal:
        return [point_tuple(index[start])]

    parent = {start: None}
    queue = deque([start])
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dx, dz in neighbors:
            nxt = (cur[0] + dx, cur[1] + dz)
            if nxt in index and nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)

    if goal not in parent:
        raise RuntimeError(
            f"no 4-connected path start={start} goal={goal}"
        )

    keys = []
    cur = goal
    while cur is not None:
        keys.append(cur)
        cur = parent[cur]
    keys.reverse()

    return [point_tuple(index[k]) for k in keys]


def angular_delta(current, target):
    return ((float(target) - float(current) + 180.0) % 360.0) - 180.0


def rotate_to_yaw(controller, target_yaw):
    pose = agent_pose(controller)
    delta = angular_delta(pose["yaw"], target_yaw)
    if abs(delta) <= 1.0:
        return True

    action = "RotateRight" if delta > 0 else "RotateLeft"
    degrees = abs(float(delta))

    ev = controller.step(
        action=action,
        degrees=degrees,
    )
    md = getattr(ev, "metadata", {}) or {}
    return bool(md.get("lastActionSuccess", True))


def face_point(controller, x, z):
    pose = agent_pose(controller)
    dx = float(x) - pose["x"]
    dz = float(z) - pose["z"]
    if math.hypot(dx, dz) <= 1e-6:
        return True

    yaw = math.degrees(math.atan2(dx, dz)) % 360.0
    # Stay on the same 30-degree motor lattice used by the embodied project.
    yaw = (round(yaw / 30.0) * 30.0) % 360.0
    return rotate_to_yaw(controller, yaw)


def desired_yaw_for_step(a, b):
    dx = float(b[0]) - float(a[0])
    dz = float(b[2]) - float(a[2])

    if abs(dx) >= abs(dz):
        return 90.0 if dx > 0 else 270.0
    return 0.0 if dz > 0 else 180.0


def execute_path(
    controller,
    path,
    *,
    move_magnitude=GRID,
    tolerance=0.13,
    stop_on_failure=True,
):
    if len(path) <= 1:
        return {
            "success": True,
            "moves": 0,
            "failed_moves": 0,
            "final_pose": agent_pose(controller),
        }

    moves = 0
    failed = 0

    for i in range(1, len(path)):
        prev = path[i - 1]
        nxt = path[i]
        target_yaw = desired_yaw_for_step(prev, nxt)
        rotate_to_yaw(controller, target_yaw)

        ev = controller.step(
            action="MoveAhead",
            moveMagnitude=float(move_magnitude),
        )
        md = getattr(ev, "metadata", {}) or {}
        ok = bool(md.get("lastActionSuccess", True))
        moves += 1

        pose = agent_pose(controller)
        err = math.hypot(
            pose["x"] - float(nxt[0]),
            pose["z"] - float(nxt[2]),
        )
        if (not ok) or err > float(tolerance):
            failed += 1
            if stop_on_failure:
                return {
                    "success": False,
                    "moves": moves,
                    "failed_moves": failed,
                    "final_pose": pose,
                    "error": str(md.get("errorMessage", "")),
                    "target_error": float(err),
                }

    return {
        "success": failed == 0,
        "moves": moves,
        "failed_moves": failed,
        "final_pose": agent_pose(controller),
    }


def teleport(controller, point, yaw=0.0):
    ev = controller.step(
        action="TeleportFull",
        x=float(point[0]),
        y=float(point[1]),
        z=float(point[2]),
        rotation={
            "x": 0.0,
            "y": float(yaw),
            "z": 0.0,
        },
        horizon=0.0,
        standing=True,
    )
    md = getattr(ev, "metadata", {}) or {}
    if not bool(md.get("lastActionSuccess", True)):
        raise RuntimeError(
            f"TeleportFull failed: {md.get('errorMessage', '')}"
        )


def centroid(points):
    return (
        sum(float(p["x"]) for p in points) / len(points),
        sum(float(p["z"]) for p in points) / len(points),
    )


def derive_door_geometry(door, closed_points, open_points, slot):
    cx, cz = centroid(closed_points)
    px = float(door["position"]["x"])
    py = float(door["position"]["y"])
    pz = float(door["position"]["z"])

    vx = px - cx
    vz = pz - cz

    if abs(vx) >= abs(vz):
        axis = "x"
        sign = 1 if vx >= 0 else -1
        label = ("east" if sign > 0 else "west") + "_door"

        approach_candidates = [
            p for p in closed_points
            if sign * (float(p["x"]) - px) < -0.05
        ]
        far_candidates = [
            p for p in open_points
            if sign * (float(p["x"]) - px) > 0.20
        ]
    else:
        axis = "z"
        sign = 1 if vz >= 0 else -1
        label = ("north" if sign > 0 else "south") + "_door"

        approach_candidates = [
            p for p in closed_points
            if sign * (float(p["z"]) - pz) < -0.05
        ]
        far_candidates = [
            p for p in open_points
            if sign * (float(p["z"]) - pz) > 0.20
        ]

    if not approach_candidates:
        raise RuntimeError(
            f"no main-side approach candidate for {door['objectId']}"
        )
    if not far_candidates:
        raise RuntimeError(
            f"no far-side candidate for {door['objectId']}"
        )

    def d2(p):
        return (
            (float(p["x"]) - px) ** 2
            + (float(p["z"]) - pz) ** 2
        )

    approach = min(approach_candidates, key=d2)

    # Prefer a point about 0.5-0.8 m beyond the threshold, not a deep-room point.
    if axis == "x":
        far = min(
            far_candidates,
            key=lambda p: (
                abs(abs(float(p["x"]) - px) - 0.50),
                abs(float(p["z"]) - pz),
                d2(p),
            ),
        )
    else:
        far = min(
            far_candidates,
            key=lambda p: (
                abs(abs(float(p["z"]) - pz) - 0.50),
                abs(float(p["x"]) - px),
                d2(p),
            ),
        )

    return DoorEpisode(
        slot=int(slot),
        label=label,
        object_id=str(door["objectId"]),
        position=(px, py, pz),
        outward_axis=axis,
        outward_sign=int(sign),
        approach=point_tuple(approach),
        far_target=point_tuple(far),
    )


def crossed_far_side(episode: DoorEpisode, pose, margin=0.20):
    if episode.outward_axis == "x":
        return (
            episode.outward_sign
            * (float(pose["x"]) - float(episode.position[0]))
            > float(margin)
        )
    return (
        episode.outward_sign
        * (float(pose["z"]) - float(episode.position[2]))
        > float(margin)
    )


@torch.no_grad()
def select_bound_episode(reader, key0, key1, payloads, query):
    slots = torch.cat(
        (key0.unsqueeze(1), key1.unsqueeze(1)),
        dim=1,
    )
    valid = torch.ones(
        slots.shape[0],
        2,
        dtype=torch.bool,
        device=slots.device,
    )
    q = torch.full(
        (slots.shape[0],),
        int(query),
        dtype=torch.long,
        device=slots.device,
    )
    out = reader(slots, valid, q)
    weights = out["attention"]
    idx = int(torch.argmax(weights, dim=-1)[0].item())
    return {
        "slot": idx,
        "payload": payloads[idx],
        "attention": [
            float(x)
            for x in weights[0].detach().cpu()
        ],
    }


def choose_start_points(closed_points, episodes, count=3):
    candidates = []
    for p in closed_points:
        dmin = min(
            math.hypot(
                float(p["x"]) - ep.position[0],
                float(p["z"]) - ep.position[2],
            )
            for ep in episodes
        )
        if dmin >= 1.25:
            candidates.append(p)

    if len(candidates) < int(count):
        candidates = list(closed_points)

    xs = [float(p["x"]) for p in candidates]
    zs = [float(p["z"]) for p in candidates]
    anchors = [
        (min(xs), min(zs)),
        (max(xs), min(zs)),
        ((min(xs) + max(xs)) / 2.0, max(zs) - 0.5),
        (min(xs), (min(zs) + max(zs)) / 2.0),
    ]

    chosen = []
    used = set()
    for ax, az in anchors:
        p = min(
            candidates,
            key=lambda q: (
                (float(q["x"]) - ax) ** 2
                + (float(q["z"]) - az) ** 2
            ),
        )
        k = qkey(p)
        if k in used:
            continue
        used.add(k)
        chosen.append(point_tuple(p))
        if len(chosen) >= int(count):
            break

    if len(chosen) < int(count):
        for p in candidates:
            k = qkey(p)
            if k not in used:
                chosen.append(point_tuple(p))
                used.add(k)
            if len(chosen) >= int(count):
                break

    return chosen
