from __future__ import annotations

from dataclasses import dataclass
import queue
import sys
import threading
from typing import Callable

from .goal_verifier import goal_satisfied
from .goals import parse_goal
from .roam import EmbodiedLoop


@dataclass
class SessionState:
    active_task: str | None = None
    quit_requested: bool = False

    @property
    def mode(self) -> str:
        return "task" if self.active_task else "autonomous"


def is_structured_verifiable_task(task: str) -> bool:
    spec = parse_goal(task)
    return bool(spec.pickup_object or spec.put_target or spec.open_object or spec.close_object)


class SessionController:
    """Pure command/state logic, separated from stdin so it is easy to test."""

    def __init__(self, initial_task: str | None = None):
        self.state = SessionState(active_task=initial_task)

    def handle(self, line: str) -> str:
        text = line.strip()
        if not text:
            return ""
        lower = text.lower()
        if lower in {"/quit", "/exit", "quit", "exit"}:
            self.state.quit_requested = True
            return "Session exit requested."
        if lower in {"/roam", "/cancel", "/free"}:
            self.state.active_task = None
            return "Returned to autonomous mode."
        if lower == "/status":
            return f"mode={self.state.mode} task={self.state.active_task or '-'}"
        if lower == "/help":
            return "Type any task to assign it. Commands: /roam /status /help /quit"
        self.state.active_task = text
        return f"Task assigned: {text}"


class InteractiveEmbodiedSession:
    """
    Keeps one AI2-THOR scene alive while autonomous behavior and user tasks alternate.

    A daemon stdin reader allows the user to type a task while an LLM request or
    movement burst is in progress. The task is applied at the next decision boundary.
    """

    def __init__(
        self,
        loop: EmbodiedLoop,
        max_decisions: int = 500,
        initial_task: str | None = None,
        input_func: Callable[[str], str] = input,
    ):
        self.loop = loop
        self.max_decisions = max_decisions
        self.controller = SessionController(initial_task=initial_task)
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
        self._reader = threading.Thread(target=self._read_commands, name="embodied-session-input", daemon=True)
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

    def run(self) -> int:
        print("Embodied session started.", flush=True)
        print("Type any sentence to assign a task while it roams.", flush=True)
        print("Commands: /roam  /status  /help  /quit", flush=True)
        self._start_reader()

        for _ in range(self.max_decisions):
            self._drain_commands()
            if self.controller.state.quit_requested:
                return 0
            if self.loop.executor.total_steps >= self.loop.executor.max_steps:
                print("[session] Primitive-action budget exhausted.", flush=True)
                return 0

            task = self.controller.state.active_task
            if task and is_structured_verifiable_task(task) and goal_satisfied(task, self.loop.thor.observe()):
                print(f"[task-complete] {task} (already satisfied in simulator)", flush=True)
                self.loop.memory.add("user_task_complete", {"task": task, "verified_by_simulator": True, "claimed_by_llm": False})
                self.controller.state.active_task = None
                task = None

            mode = "task" if task else "autonomous"
            decision, _record = self.loop.step(mode=mode, task=task)

            if task:
                world_after = self.loop.thor.observe()
                verified = is_structured_verifiable_task(task) and goal_satisfied(task, world_after)
                claimed = decision.task_complete and not is_structured_verifiable_task(task)
                if verified or claimed:
                    print(f"[task-complete] {task}", flush=True)
                    self.loop.memory.add(
                        "user_task_complete",
                        {"task": task, "verified_by_simulator": bool(verified), "claimed_by_llm": bool(claimed)},
                    )
                    self.controller.state.active_task = None
                    print("[session] Returned to autonomous mode.", flush=True)

        print("[session] Decision budget exhausted.", flush=True)
        return 0
