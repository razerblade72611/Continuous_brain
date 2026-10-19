#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import random

START = 0
LEFT_MID = 1
RIGHT_MID = 2
LEFT_GOAL = 3
RIGHT_GOAL = 4
FAIL = 5
NODE_NAMES = (
    "start",
    "left_mid",
    "right_mid",
    "left_goal",
    "right_goal",
    "fail",
)

APPROACH_LEFT = 0
APPROACH_RIGHT = 1
DIRECT = 2
SWITCH = 3
CROSS_FINISH = 4
ACTION_NAMES = (
    "approach_left",
    "approach_right",
    "direct",
    "switch",
    "cross_finish",
)

OPEN = 0
KNOWN_BLOCK = 1
SURPRISE_BLOCK = 2
MODE_NAMES = ("open", "known_block", "surprise_block")


@dataclass(frozen=True)
class RouteSpec:
    condition_id: int
    mode: int
    approach_left_cost: float
    approach_right_cost: float
    direct_cost: float
    switch_cost: float
    cross_cost: float


def generate_spec(condition_id: int) -> RouteSpec:
    """
    Infinite deterministic family.  Condition ranges are assigned by protocol.

    mode cycles across open / known-block / surprise-block while costs vary
    independently enough that the planner cannot simply memorize one scalar
    return per action.
    """
    cid = int(condition_id)
    rng = random.Random(0xC025A + 7919 * cid)
    mode = cid % 3

    return RouteSpec(
        condition_id=cid,
        mode=mode,
        approach_left_cost=0.90 + 0.20 * rng.random(),
        approach_right_cost=0.90 + 0.20 * rng.random(),
        direct_cost=0.55 + 0.20 * rng.random(),
        switch_cost=0.55 + 0.20 * rng.random(),
        cross_cost=1.20 + 0.25 * rng.random(),
    )


class DynamicRouteWorld:
    """
    Small new planning/replanning family.

    Goal is external and never encoded in route state.

    OPEN:
      target-side approach -> DIRECT is optimal.

    KNOWN_BLOCK:
      start state exposes known_target_blocked=1.
      Optimal plan approaches the opposite lane and CROSS_FINISHes to the goal.

    SURPRISE_BLOCK:
      start looks exactly like OPEN.
      The model therefore predicts target-side approach -> unblocked mid.
      After that approach, reality changes: the target lane is blocked.
      A controller that compares prediction to observation can replan:
          target_mid(blocked) -> SWITCH -> opposite_mid -> CROSS_FINISH -> goal.
      A plan-once controller that executes stale DIRECT fails.
    """
    def __init__(self, condition_id: int, goal_side: int):
        self.spec = generate_spec(condition_id)
        self.goal_side = int(goal_side)
        if self.goal_side not in (0, 1):
            raise ValueError("goal_side must be 0/1")
        self.reset()

    def reset(self):
        self.node = START
        self.blocked = False
        self.done = False
        self.success = False
        self.total_cost = 0.0
        self.steps = 0
        self.surprise_triggered = False
        return self.state()

    @property
    def known_target_blocked(self):
        return self.spec.mode == KNOWN_BLOCK

    def state(self):
        return {
            "node": int(self.node),
            "blocked": bool(self.blocked),
            "known_target_blocked": bool(self.known_target_blocked),
        }

    def valid_actions(self):
        if self.done:
            return []
        if self.node == START:
            return [APPROACH_LEFT, APPROACH_RIGHT]
        if self.node in (LEFT_MID, RIGHT_MID):
            return [DIRECT, SWITCH, CROSS_FINISH]
        return []

    def _lane_side(self):
        if self.node == LEFT_MID:
            return 0
        if self.node == RIGHT_MID:
            return 1
        return None

    def _cost(self, action: int):
        if action == APPROACH_LEFT:
            return self.spec.approach_left_cost
        if action == APPROACH_RIGHT:
            return self.spec.approach_right_cost
        if action == DIRECT:
            return self.spec.direct_cost
        if action == SWITCH:
            return self.spec.switch_cost
        if action == CROSS_FINISH:
            return self.spec.cross_cost
        raise ValueError(action)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode already done")
        action = int(action)
        if action not in self.valid_actions():
            self.node = FAIL
            self.done = True
            self.success = False
            self.steps += 1
            self.total_cost += 3.0
            return self.state(), -3.0, True, self.info()

        cost = float(self._cost(action))
        self.total_cost += cost
        self.steps += 1

        if self.node == START:
            lane = 0 if action == APPROACH_LEFT else 1
            self.node = LEFT_MID if lane == 0 else RIGHT_MID

            # Known block is predictable from start.  Surprise is intentionally
            # not present in the start state and appears only after committing
            # to the target-side approach.
            if self.spec.mode == KNOWN_BLOCK:
                self.blocked = bool(lane == self.goal_side)
            elif self.spec.mode == SURPRISE_BLOCK and lane == self.goal_side:
                self.blocked = True
                self.surprise_triggered = True
            else:
                self.blocked = False

        elif self.node in (LEFT_MID, RIGHT_MID):
            lane = self._lane_side()

            if action == DIRECT:
                if self.blocked:
                    self.node = FAIL
                    self.done = True
                    self.success = False
                else:
                    self.node = LEFT_GOAL if lane == 0 else RIGHT_GOAL
                    self.done = True
                    self.success = bool(lane == self.goal_side)

            elif action == SWITCH:
                self.node = RIGHT_MID if lane == 0 else LEFT_MID
                # Alternate lane is the bypass lane and is open.
                self.blocked = False

            elif action == CROSS_FINISH:
                destination = 1 - lane
                self.node = LEFT_GOAL if destination == 0 else RIGHT_GOAL
                self.done = True
                self.success = bool(destination == self.goal_side)

        reward = (10.0 if self.success else 0.0) - cost
        return self.state(), reward, self.done, self.info()

    def info(self):
        return {
            "success": bool(self.success),
            "node": int(self.node),
            "node_name": NODE_NAMES[int(self.node)],
            "steps": int(self.steps),
            "total_cost": float(self.total_cost),
            "mode": MODE_NAMES[int(self.spec.mode)],
            "surprise_triggered": bool(self.surprise_triggered),
        }


def model_transition(state, goal_side: int, action: int, spec: RouteSpec):
    """
    Training transition model: all dynamics except SURPRISE exogenous change.

    This function intentionally treats SURPRISE_BLOCK as OPEN at START.
    Thus the learned consequence model has a falsifiable prediction that can
    be violated by the real surprise environment.
    """
    node = int(state["node"])
    blocked = bool(state["blocked"])
    known = bool(state["known_target_blocked"])
    goal_side = int(goal_side)
    action = int(action)

    valid = (
        (node == START and action in (APPROACH_LEFT, APPROACH_RIGHT))
        or (
            node in (LEFT_MID, RIGHT_MID)
            and action in (DIRECT, SWITCH, CROSS_FINISH)
        )
    )
    if not valid:
        return {
            "next_node": FAIL,
            "next_blocked": False,
            "cost": 3.0,
            "success": False,
            "valid": False,
        }

    if action == APPROACH_LEFT:
        cost = spec.approach_left_cost
    elif action == APPROACH_RIGHT:
        cost = spec.approach_right_cost
    elif action == DIRECT:
        cost = spec.direct_cost
    elif action == SWITCH:
        cost = spec.switch_cost
    else:
        cost = spec.cross_cost

    if node == START:
        lane = 0 if action == APPROACH_LEFT else 1
        next_node = LEFT_MID if lane == 0 else RIGHT_MID
        next_blocked = bool(known and lane == goal_side)
        return {
            "next_node": next_node,
            "next_blocked": next_blocked,
            "cost": float(cost),
            "success": False,
            "valid": True,
        }

    lane = 0 if node == LEFT_MID else 1

    if action == DIRECT:
        if blocked:
            return {
                "next_node": FAIL,
                "next_blocked": False,
                "cost": float(cost),
                "success": False,
                "valid": True,
            }
        next_node = LEFT_GOAL if lane == 0 else RIGHT_GOAL
        return {
            "next_node": next_node,
            "next_blocked": False,
            "cost": float(cost),
            "success": bool(lane == goal_side),
            "valid": True,
        }

    if action == SWITCH:
        return {
            "next_node": RIGHT_MID if lane == 0 else LEFT_MID,
            "next_blocked": False,
            "cost": float(cost),
            "success": False,
            "valid": True,
        }

    destination = 1 - lane
    return {
        "next_node": LEFT_GOAL if destination == 0 else RIGHT_GOAL,
        "next_blocked": False,
        "cost": float(cost),
        "success": bool(destination == goal_side),
        "valid": True,
    }
