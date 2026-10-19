from __future__ import annotations

from .goals import GoalSpec, parse_goal
from .models import WorldState


def goal_satisfied(goal: str | GoalSpec, world: WorldState) -> bool:
    spec = parse_goal(goal) if isinstance(goal, str) else goal

    # A placement goal outranks the pickup clause of a composite task.
    if spec.has_put:
        source_name = spec.source_object
        target_name = spec.put_target
        if not source_name or not target_name:
            return False

        sources = world.find(source_name)
        targets = world.find(target_name)
        if not sources or not targets:
            return False

        target_ids = {obj.object_id for obj in targets}
        return any(
            any(parent_id in target_ids for parent_id in source.parent_receptacles)
            for source in sources
        )

    if spec.pickup_object:
        return any(obj.is_picked_up for obj in world.find(spec.pickup_object))

    if spec.open_object:
        return any(obj.is_open is True for obj in world.find(spec.open_object))

    if spec.close_object:
        return any(obj.is_open is False for obj in world.find(spec.close_object))

    return False
