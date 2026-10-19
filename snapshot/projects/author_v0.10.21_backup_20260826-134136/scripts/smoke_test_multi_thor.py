import argparse

from agent_core.thor_adapter import SharedThorWorld, ThorConfig
from agent_core.models import Action


def main():
    parser = argparse.ArgumentParser(description="Two-agent AI2-THOR smoke test")
    parser.add_argument("--thor-executable", default=None, help="Optional custom AI2-THOR local player executable")
    parser.add_argument("--scene", default="FloorPlan1")
    args = parser.parse_args()

    import ai2thor
    print("AI2-THOR:", ai2thor.__version__)
    world = SharedThorWorld(
        ThorConfig(
            scene=args.scene,
            width=640,
            height=480,
            render_instance_segmentation=True,
            local_executable_path=args.thor_executable,
        ),
        agent_count=2,
    )
    world.start()
    try:
        a = world.agent(0)
        b = world.agent(1)
        a.start(); b.start()
        sa = a.observe(); sb = b.observe()
        print("Shared controller started")
        print("Custom executable:", args.thor_executable or "stock AI2-THOR build")
        print("Agent A pose:", sa.pose.position, sa.pose.rotation)
        print("Agent B pose:", sb.pose.position, sb.pose.rotation)
        print("Agent A frame:", a.raw_event().frame.shape if a.raw_event().frame is not None else None)
        print("Agent B frame:", b.raw_event().frame.shape if b.raw_event().frame is not None else None)
        before_a = float(sa.pose.rotation.get("y", 0.0))
        before_b = float(sb.pose.rotation.get("y", 0.0))
        b.execute(Action("RotateRight", {}, "multi-agent smoke test"))
        after_a = float(a.observe().pose.rotation.get("y", 0.0))
        after_b = float(b.observe().pose.rotation.get("y", 0.0))
        print("Rotate Agent B success:", b.observe().last_action_success)
        print("A yaw before/after:", before_a, after_a)
        print("B yaw before/after:", before_b, after_b)
        print("Multi-agent smoke test complete")
    finally:
        world.close()


if __name__ == "__main__":
    main()
