#!/usr/bin/env python3
from continuous_brain.dynamic_route_world_v1 import (
    OPEN,
    KNOWN_BLOCK,
    SURPRISE_BLOCK,
    APPROACH_LEFT,
    APPROACH_RIGHT,
    DIRECT,
    SWITCH,
    CROSS_FINISH,
    DynamicRouteWorld,
    generate_spec,
    model_transition,
)


def find_condition(mode):
    for cid in range(20):
        if generate_spec(cid).mode == mode:
            return cid
    raise AssertionError(mode)


def main():
    # OPEN: target-side direct route works.
    cid=find_condition(OPEN)
    env=DynamicRouteWorld(cid,0)
    s=env.reset()
    assert s["known_target_blocked"] is False
    env.step(APPROACH_LEFT)
    assert env.state()["blocked"] is False
    env.step(DIRECT)
    assert env.success is True

    # KNOWN_BLOCK: start exposes block; opposite lane + cross succeeds.
    cid=find_condition(KNOWN_BLOCK)
    env=DynamicRouteWorld(cid,0)
    s=env.reset()
    assert s["known_target_blocked"] is True
    env.step(APPROACH_RIGHT)
    assert env.state()["blocked"] is False
    env.step(CROSS_FINISH)
    assert env.success is True

    # SURPRISE: model transition predicts unblocked, reality becomes blocked.
    cid=find_condition(SURPRISE_BLOCK)
    env=DynamicRouteWorld(cid,0)
    s=env.reset()
    spec=env.spec
    pred=model_transition(s,0,APPROACH_LEFT,spec)
    assert pred["next_blocked"] is False
    actual,_,done,_=env.step(APPROACH_LEFT)
    assert done is False
    assert actual["blocked"] is True
    assert env.surprise_triggered is True

    # Stale direct fails.
    env.step(DIRECT)
    assert env.success is False

    # Fresh surprise episode can recover after mismatch.
    env=DynamicRouteWorld(cid,0)
    env.reset()
    env.step(APPROACH_LEFT)
    env.step(SWITCH)
    env.step(CROSS_FINISH)
    assert env.success is True

    print("CONSEQUENCE_REPLANNING_WORLD_SMOKE: PASS")


if __name__=="__main__":
    main()
