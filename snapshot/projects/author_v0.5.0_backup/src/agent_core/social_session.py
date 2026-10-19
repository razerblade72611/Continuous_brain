from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
from typing import Callable, Mapping

from .goal_verifier import goal_satisfied
from .session import is_structured_verifiable_task
from .roam import EmbodiedLoop
from .social import SpeechBus


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
    ):
        self.loops = dict(loops)
        self.labels = dict(labels)
        self.speech_bus = speech_bus
        self.max_rounds = int(max_rounds)
        self.controller = SocialSessionController(self.labels)
        self.input_func = input_func
        self.commands: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None

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
        loop.memory.add(
            "user_task_complete",
            {"task": task, "verified_by_simulator": bool(verified), "claimed_by_llm": bool(claimed and not structured)},
        )
        self.controller.state.tasks[agent_id] = None
        print(f"[{label}] Returned to autonomous mode.", flush=True)

    def run(self) -> int:
        names = ", ".join(self.labels[i] for i in sorted(self.labels))
        print(f"Social embodied session started with agents: {names}.", flush=True)
        print("Each agent has its own camera, memory, focus, task state, and LLM context.", flush=True)
        print("Speech is explicit and range-limited; private agent state is never copied between agents.", flush=True)
        print("Commands: /a <task>  /b <task>  /both <task>  /a roam  /b roam  /both roam  /status  /quit", flush=True)
        self._start_reader()

        for _round in range(1, self.max_rounds + 1):
            for agent_id in sorted(self.loops):
                self._drain_commands()
                if self.controller.state.quit_requested:
                    return 0
                loop = self.loops[agent_id]
                label = self.labels[agent_id]
                if loop.executor.total_steps >= loop.executor.max_steps:
                    print(f"[{label}] Primitive-action budget exhausted; this body will remain idle.", flush=True)
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

                social_context = {
                    "self_label": label,
                    "heard_since_last_turn": [event.for_llm() for event in heard],
                    "speech_range_m": self.speech_bus.speech_range_m,
                }
                mode = "task" if task else "autonomous"
                decision, _record = loop.step(
                    mode=mode,
                    task=task,
                    social_context=social_context,
                    print_prefix=f"{label}-{mode}",
                )

                for utterance in decision.spoken or []:
                    loop.memory.add("spoken_speech", {"text": utterance})
                    delivered = self.speech_bus.speak(agent_id, utterance, self._worlds())
                    print(f"[{label} says] {utterance}", flush=True)
                    if not delivered:
                        print(f"[{label} speech] No other agent was within hearing range.", flush=True)

                if task:
                    self._complete_task_if_needed(agent_id, claimed=decision.task_complete)

        print("[social-session] Round budget exhausted.", flush=True)
        return 0
