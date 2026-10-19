from __future__ import annotations

import ai2thor
from ai2thor.controller import Controller


print("AI2-THOR:", ai2thor.__version__)
controller = Controller(scene="FloorPlan1", width=640, height=480)
try:
    print("Controller started")
    print("Scene:", controller.last_event.metadata.get("sceneName"))
    print("Objects:", len(controller.last_event.metadata.get("objects", [])))
    event = controller.step(action="RotateRight")
    print("RotateRight success:", event.metadata.get("lastActionSuccess"))
    print("Error:", repr(event.metadata.get("errorMessage")))
finally:
    controller.stop()
    print("Controller stopped cleanly")
