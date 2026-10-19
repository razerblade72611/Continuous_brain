#!/usr/bin/env python3

import argparse
import math
from ai2thor.controller import Controller


def yaw_toward(agent_pos, target_pos):
    dx = target_pos["x"] - agent_pos["x"]
    dz = target_pos["z"] - agent_pos["z"]
    return math.degrees(math.atan2(dx, dz)) % 360.0


def find_object(event, object_id):
    for obj in event.metadata.get("objects", []):
        if obj.get("objectId") == object_id:
            return obj
    return None


def inventory_ids(event):
    return {
        obj.get("objectId")
        for obj in event.metadata.get("inventoryObjects", [])
    }


def get_reachable(controller):
    event = controller.step(action="GetReachablePositions")
    if not event.metadata.get("lastActionSuccess"):
        raise RuntimeError(
            "GetReachablePositions failed: "
            + event.metadata.get("errorMessage", "")
        )
    return event.metadata.get("actionReturn", [])


def get_candidate_positions(reachable, target_pos):
    scored = []

    for p in reachable:
        dx = p["x"] - target_pos["x"]
        dz = p["z"] - target_pos["z"]
        distance = math.sqrt(dx * dx + dz * dz)

        # Prefer approximately arm's-length positions.
        if 0.45 <= distance <= 1.35:
            score = abs(distance - 0.85)
            scored.append((score, distance, p))

    if not scored:
        for p in reachable:
            dx = p["x"] - target_pos["x"]
            dz = p["z"] - target_pos["z"]
            distance = math.sqrt(dx * dx + dz * dz)
            scored.append((distance, distance, p))

    scored.sort(key=lambda item: item[0])
    return [item[2] for item in scored[:30]]


def place_and_face(controller, object_id, reachable):
    current = find_object(controller.last_event, object_id)
    if current is None:
        return None

    target_pos = current["position"]
    candidate_positions = get_candidate_positions(
        reachable,
        target_pos,
    )

    # Positive horizon looks downward in AI2-THOR.
    horizons = [30, 0, 45, -30]

    for point in candidate_positions:
        yaw = yaw_toward(point, target_pos)

        for horizon in horizons:
            event = controller.step(
                action="TeleportFull",
                x=point["x"],
                y=point["y"],
                z=point["z"],
                rotation={
                    "x": 0,
                    "y": yaw,
                    "z": 0,
                },
                horizon=horizon,
                standing=True,
            )

            if not event.metadata.get("lastActionSuccess"):
                continue

            obj = find_object(event, object_id)

            if obj is not None and obj.get("visible"):
                return obj

    return None


def test_pickup_drop(controller):
    objects = controller.last_event.metadata.get("objects", [])

    pickupables = [
        obj for obj in objects
        if obj.get("pickupable")
    ]

    # Prefer objects that are not hidden inside receptacles.
    pickupables.sort(
        key=lambda obj: (
            bool(obj.get("parentReceptacles")),
            obj.get("objectType", ""),
        )
    )

    print(f"Found {len(pickupables)} pickupable objects in scene metadata.")

    reachable = get_reachable(controller)
    print(f"Found {len(reachable)} reachable agent positions.")

    failures = []

    for candidate in pickupables:
        object_id = candidate["objectId"]
        object_type = candidate.get("objectType", object_id)

        print()
        print(f"Trying pickup candidate: {object_type}")
        print(f"  id: {object_id}")

        visible_obj = place_and_face(
            controller,
            object_id,
            reachable,
        )

        if visible_obj is None:
            failures.append(
                f"{object_type}: could not obtain visible reachable pose"
            )
            continue

        print("  visible near agent: YES")

        event = controller.step(
            action="PickupObject",
            objectId=object_id,
        )

        if not event.metadata.get("lastActionSuccess"):
            error = event.metadata.get("errorMessage", "")
            print(f"  PickupObject failed: {error}")
            failures.append(
                f"{object_type}: PickupObject failed: {error}"
            )
            continue

        held = object_id in inventory_ids(event)

        print(f"  PickupObject success: {event.metadata.get('lastActionSuccess')}")
        print(f"  verified in inventory: {held}")

        if not held:
            failures.append(
                f"{object_type}: simulator reported success but "
                "object not present in inventory"
            )
            continue

        drop_event = controller.step(
            action="DropHandObject",
        )

        drop_ok = (
            drop_event.metadata.get("lastActionSuccess")
            and object_id not in inventory_ids(drop_event)
        )

        print(
            f"  DropHandObject success: "
            f"{drop_event.metadata.get('lastActionSuccess')}"
        )
        print(f"  verified hand empty: {drop_ok}")

        if not drop_ok:
            failures.append(
                f"{object_type}: drop verification failed: "
                + drop_event.metadata.get("errorMessage", "")
            )
            continue

        print()
        print(f"PASS pickup/drop: {object_type}")
        return True

    print()
    print("Pickup candidates attempted but none completed successfully.")

    for failure in failures[:20]:
        print("  -", failure)

    return False


def test_openable(controller):
    objects = controller.last_event.metadata.get("objects", [])

    candidates = [
        obj for obj in objects
        if obj.get("openable") and not obj.get("isOpen")
    ]

    if not candidates:
        print("SKIP OpenObject: no closed openable objects found.")
        return

    reachable = get_reachable(controller)

    for candidate in candidates:
        object_id = candidate["objectId"]
        object_type = candidate.get("objectType", object_id)

        visible_obj = place_and_face(
            controller,
            object_id,
            reachable,
        )

        if visible_obj is None:
            continue

        event = controller.step(
            action="OpenObject",
            objectId=object_id,
        )

        if not event.metadata.get("lastActionSuccess"):
            continue

        after = find_object(event, object_id)

        if after and after.get("isOpen"):
            print(f"PASS OpenObject: {object_type}")

            # Restore scene if possible.
            controller.step(
                action="CloseObject",
                objectId=object_id,
            )
            return

    print("SKIP OpenObject: no reachable visible candidate succeeded.")


def test_toggleable(controller):
    objects = controller.last_event.metadata.get("objects", [])

    candidates = [
        obj for obj in objects
        if obj.get("toggleable") and not obj.get("isToggled")
    ]

    if not candidates:
        print("SKIP ToggleObjectOn: no off toggleable objects found.")
        return

    reachable = get_reachable(controller)

    for candidate in candidates:
        object_id = candidate["objectId"]
        object_type = candidate.get("objectType", object_id)

        visible_obj = place_and_face(
            controller,
            object_id,
            reachable,
        )

        if visible_obj is None:
            continue

        event = controller.step(
            action="ToggleObjectOn",
            objectId=object_id,
        )

        if not event.metadata.get("lastActionSuccess"):
            continue

        after = find_object(event, object_id)

        if after and after.get("isToggled"):
            print(f"PASS ToggleObjectOn: {object_type}")

            controller.step(
                action="ToggleObjectOff",
                objectId=object_id,
            )
            return

    print("SKIP ToggleObjectOn: no reachable visible candidate succeeded.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        default="FloorPlan1",
    )

    parser.add_argument(
        "--thor-executable",
        required=True,
    )

    args = parser.parse_args()

    print("Starting custom AI2-THOR interaction test...")
    print("Scene:", args.scene)
    print("Executable:", args.thor_executable)
    print()

    controller = Controller(
        scene=args.scene,
        local_executable_path=args.thor_executable,
        width=640,
        height=480,
        visibilityDistance=1.5,
    )

    try:
        print(
            "Loaded scene:",
            controller.last_event.metadata.get("sceneName"),
        )

        print()
        print("=== PICKUP / DROP TEST ===")

        pickup_ok = test_pickup_drop(controller)

        print()
        print("=== OPEN / CLOSE TEST ===")

        test_openable(controller)

        print()
        print("=== TOGGLE TEST ===")

        test_toggleable(controller)

        print()
        print("=== RESULT ===")

        if not pickup_ok:
            raise RuntimeError(
                "Could not complete a native PickupObject/DropHandObject "
                "cycle anywhere in the scene."
            )

        print("PASS: grounded native manipulation is operational.")

    finally:
        controller.stop()


if __name__ == "__main__":
    main()
