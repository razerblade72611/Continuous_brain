from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
from typing import Any, Callable, Mapping

from .goal_verifier import goal_satisfied
from .session import is_structured_verifiable_task
from .roam import EmbodiedLoop
from .social import SpeechBus, peer_presence_for


@dataclass
class SocialControlState:
    tasks: dict[int, str | None] = field(default_factory=lambda: {0: None, 1: None})
    quit_requested: bool = False


class SocialSessionController:
    """Routes user commands to A, B, or both without changing autonomous preferences."""

    def __init__(self, labels: Mapping[int, str] | None = None):
        self.labels = dict(labels or {0: "A", 1: "B"})
        self.state = SocialControlState(tasks={agent_id: None for agent_id in self.labels})

    def _ids_for_prefix(self, prefix: str) -> list[int]:
        p = prefix.lower()
        if p in {"both", "all"}:
            return list(sorted(self.labels))
        for agent_id, label in self.labels.items():
            if p == str(label).lower() or p == str(agent_id):
                return [agent_id]
        return []

    def handle(self, line: str) -> str:
        text = line.strip()
        if not text:
            return ""
        low = text.lower()
        if low in {"/quit", "/exit", "quit", "exit"}:
            self.state.quit_requested = True
            return "Social session exit requested."
        if low == "/status":
            parts = []
            for agent_id in sorted(self.labels):
                task = self.state.tasks.get(agent_id)
                parts.append(f"{self.labels[agent_id]}={'task: ' + task if task else 'autonomous'}")
            return " | ".join(parts)
        if low == "/help":
            return "Commands: /a <task>, /b <task>, /both <task>, /a roam, /b roam, /both roam, /status, /quit. Plain text assigns the same task to both."

        if text.startswith("/"):
            head, _, body = text[1:].partition(" ")
            ids = self._ids_for_prefix(head)
            if not ids:
                return f"Unknown agent route '/{head}'. Use /a, /b, or /both."
            body = body.strip()
            if not body:
                return f"No task supplied for /{head}."
            if body.lower() in {"roam", "free", "cancel"}:
                for agent_id in ids:
                    self.state.tasks[agent_id] = None
                return f"Returned {', '.join(self.labels[i] for i in ids)} to autonomous mode."
            for agent_id in ids:
                self.state.tasks[agent_id] = body
            return f"Task assigned to {', '.join(self.labels[i] for i in ids)}: {body}"

        for agent_id in self.labels:
            self.state.tasks[agent_id] = text
        return f"Shared task assigned to all agents: {text}"


class SocialEmbodiedSession:
    """
    Two independent embodied LLM contexts alternating inside one shared Unity scene.

    The coordinator serializes inference. The only cross-agent information it moves is
    explicit speech that passes the physical hearing-range check.
    """

    def __init__(
        self,
        loops: Mapping[int, EmbodiedLoop],
        labels: Mapping[int, str],
        speech_bus: SpeechBus,
        max_rounds: int = 500,
        input_func: Callable[[str], str] = input,
        dashboard: Any | None = None,
        peer_sensor_enabled: bool = True,
        peer_camera_cone_margin_degrees: float = 3.0,
    ):
        self.loops = dict(loops)
        self.labels = dict(labels)
        self.speech_bus = speech_bus
        self.max_rounds = int(max_rounds)
        self.controller = SocialSessionController(self.labels)
        self.input_func = input_func
        self.commands: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.dashboard = dashboard
        self.peer_sensor_enabled = bool(peer_sensor_enabled)
        self.peer_camera_cone_margin_degrees = float(peer_camera_cone_margin_degrees)

    def _read_commands(self) -> None:
        while not self.controller.state.quit_requested:
            try:
                line = self.input_func("")
            except (EOFError, KeyboardInterrupt):
                return
            self.commands.put(line)

    def _start_reader(self) -> None:
        self._reader = threading.Thread(target=self._read_commands, name="social-session-input", daemon=True)
        self._reader.start()

    def _drain_commands(self) -> None:
        while True:
            try:
                line = self.commands.get_nowait()
            except queue.Empty:
                return
            message = self.controller.handle(line)
            if message:
                print(f"[user-control] {message}", flush=True)
                if self.dashboard is not None:
                    self.dashboard.append_event("user-control", message)
                    for agent_id, task in self.controller.state.tasks.items():
                        self.dashboard.set_task_state(agent_id, task)

    def _worlds(self):
        return {agent_id: loop.thor.observe() for agent_id, loop in self.loops.items()}

    def _complete_task_if_needed(self, agent_id: int, claimed: bool = False) -> None:
        task = self.controller.state.tasks.get(agent_id)
        if not task:
            return
        loop = self.loops[agent_id]
        structured = is_structured_verifiable_task(task)
        verified = structured and goal_satisfied(task, loop.thor.observe())
        done = bool(verified or (claimed and not structured))
        if not done:
            return
        label = self.labels[agent_id]
        print(f"[{label} task-complete] {task}", flush=True)
        if self.dashboard is not None:
            self.dashboard.append_event("task-complete", task, agent_id=agent_id)
        loop.memory.add(
            "user_task_complete",
            {"task": task, "verified_by_simulator": bool(verified), "claimed_by_llm": bool(claimed and not structured)},
        )
        self.controller.state.tasks[agent_id] = None
        if self.dashboard is not None:
            self.dashboard.set_task_state(agent_id, None)
        print(f"[{label}] Returned to autonomous mode.", flush=True)

    def run(self) -> int:
        names = ", ".join(self.labels[i] for i in sorted(self.labels))
        print(f"Social embodied session started with agents: {names}.", flush=True)
        print("Each agent has its own camera, memory, focus, task state, and LLM context.", flush=True)
        print("Speech is explicit and range-limited; private agent state is never copied between agents.", flush=True)
        print("Commands: /a <task>  /b <task>  /both <task>  /a roam  /b roam  /both roam  /status  /quit", flush=True)

        if self.dashboard is not None:
            self.dashboard.start()
            self.dashboard.refresh_frames(self.loops)
            self.dashboard.append_event("session", f"Social embodied session started with agents: {names}.")
            print(f"Observer dashboard: {self.dashboard.url}", flush=True)
            print("Dashboard is read-only; it receives reports after decisions and never feeds state back to either agent.", flush=True)

        self._start_reader()
        try:
            for _round in range(1, self.max_rounds + 1):
                for agent_id in sorted(self.loops):
                    self._drain_commands()
                    if self.controller.state.quit_requested:
                        return 0
                    loop = self.loops[agent_id]
                    label = self.labels[agent_id]
                    if loop.executor.total_steps >= loop.executor.max_steps:
                        print(f"[{label}] Primitive-action budget exhausted; this body will remain idle.", flush=True)
                        if self.dashboard is not None:
                            self.dashboard.append_event("budget", "Primitive-action budget exhausted; body idle.", agent_id=agent_id)
                            self.dashboard.refresh_frames(self.loops)
                        continue

                    task = self.controller.state.tasks.get(agent_id)
                    if task and is_structured_verifiable_task(task) and goal_satisfied(task, loop.thor.observe()):
                        self._complete_task_if_needed(agent_id, claimed=False)
                        task = None

                    heard = self.speech_bus.consume(agent_id)
                    if heard:
                        for event in heard:
                            loop.memory.add("heard_speech", event.for_llm())
                            print(f"[{label} hears {event.speaker_label}] {event.text}", flush=True)
                            if self.dashboard is not None:
                                self.dashboard.append_event(
                                    "speech",
                                    f"heard {event.speaker_label}: {event.text}",
                                    agent_id=agent_id,
                                    peer_id=event.speaker_id,
                                )

                    worlds_before = self._worlds()
                    peer_presence = peer_presence_for(
                        agent_id,
                        worlds_before,
                        self.labels,
                        field_of_view_degrees=float(loop.thor.config.field_of_view),
                        speech_range_m=self.speech_bus.speech_range_m,
                        camera_cone_margin_degrees=self.peer_camera_cone_margin_degrees,
                    )
                    model_peer_sensor = peer_presence if self.peer_sensor_enabled else []
                    social_context = {
                        "self_label": label,
                        "heard_since_last_turn": [event.for_llm() for event in heard],
                        "speech_range_m": self.speech_bus.speech_range_m,
                        # Geometry-only peer sensor. It does not identify scene objects or
                        # claim pixel-level visibility/occlusion; RGB remains perceptual input.
                        "peer_camera_cone_sensor": [item.for_llm() for item in model_peer_sensor],
                    }
                    mode = "task" if task else "autonomous"
                    decision, record = loop.step(
                        mode=mode,
                        task=task,
                        social_context=social_context,
                        print_prefix=f"{label}-{mode}",
                    )

                    if self.dashboard is not None:
                        worlds_after = self._worlds()
                        dashboard_peer_presence = peer_presence_for(
                            agent_id,
                            worlds_after,
                            self.labels,
                            field_of_view_degrees=float(loop.thor.config.field_of_view),
                            speech_range_m=self.speech_bus.speech_range_m,
                            camera_cone_margin_degrees=self.peer_camera_cone_margin_degrees,
                        )
                        # Keep peer geometry for BOTH observer panels synchronized with
                        # the same shared-world state after every embodied action.
                        for observer_id, observer_loop in self.loops.items():
                            current_presence = peer_presence_for(
                                observer_id,
                                worlds_after,
                                self.labels,
                                field_of_view_degrees=float(observer_loop.thor.config.field_of_view),
                                speech_range_m=self.speech_bus.speech_range_m,
                                camera_cone_margin_degrees=self.peer_camera_cone_margin_degrees,
                            )
                            self.dashboard.set_peer_presence(observer_id, current_presence)
                        self.dashboard.update_agent(
                            agent_id,
                            loop,
                            mode=mode,
                            task=task,
                            decision=decision,
                            record=record,
                            heard=heard,
                            peer_presence=dashboard_peer_presence,
                        )
                        # Refresh both cameras after every embodied turn so the two
                        # side-by-side panels represent the same current shared world.
                        self.dashboard.refresh_frames(self.loops)

                    for utterance in decision.spoken or []:
                        loop.memory.add("spoken_speech", {"text": utterance})
                        delivered = self.speech_bus.speak(agent_id, utterance, self._worlds())
                        print(f"[{label} says] {utterance}", flush=True)
                        if self.dashboard is not None:
                            self.dashboard.append_event("speech", f"said: {utterance}", agent_id=agent_id)
                        if not delivered:
                            print(f"[{label} speech] No other agent was within hearing range.", flush=True)
                            if self.dashboard is not None:
                                self.dashboard.append_event(
                                    "speech",
                                    "No other agent was within hearing range.",
                                    agent_id=agent_id,
                                )

                    if task:
                        self._complete_task_if_needed(agent_id, claimed=decision.task_complete)

            print("[social-session] Round budget exhausted.", flush=True)
            if self.dashboard is not None:
                self.dashboard.append_event("session", "Round budget exhausted.")
            return 0
        finally:
            if self.dashboard is not None:
                self.dashboard.stop()
