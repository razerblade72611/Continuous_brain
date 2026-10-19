from __future__ import annotations

from dataclasses import dataclass

from .executor import ActionExecutor
from .goal_verifier import goal_satisfied
from .memory import EpisodicMemory
from .planner import HybridPlanner
from .recovery import RecoveryManager
from .thor_adapter import ThorAdapter


@dataclass
class AgentResult:
    success: bool
    goal: str
    reason: str
    steps: int


class Agent:
    def __init__(
        self,
        thor: ThorAdapter,
        planner: HybridPlanner,
        executor: ActionExecutor,
        memory: EpisodicMemory,
        recovery: RecoveryManager,
        max_replans: int = 20,
    ):
        self.thor = thor
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.recovery = recovery
        self.max_replans = max_replans

    def run(self, goal: str) -> AgentResult:
        recent_context: list[dict] = []

        for replan_count in range(self.max_replans):
            world = self.thor.observe()
            if goal_satisfied(goal, world):
                return AgentResult(True, goal, "Goal verified from current world metadata.", self.executor.total_steps)

            plan = self.planner.plan(goal, world, recent_context)
            self.memory.add(
                "plan",
                {
                    "goal": goal,
                    "rationale": plan.rationale,
                    "actions": [action.as_dict() for action in plan.actions],
                    "replan": replan_count,
                },
            )

            if not plan.actions:
                if goal_satisfied(goal, self.thor.observe()):
                    return AgentResult(True, goal, "Goal verified after an empty plan.", self.executor.total_steps)
                recent_context.append({"event": "empty_plan", "replan": replan_count})
                recent_context = recent_context[-12:]
                continue

            failed = False
            for action in plan.actions:
                outcome = self.executor.execute(action)
                recent_context.append(
                    {
                        "action": action.as_dict(),
                        "success": outcome.success,
                        "error": outcome.error,
                        "verification": outcome.verification,
                        "primitive_steps": outcome.primitive_steps,
                    }
                )
                recent_context = recent_context[-12:]

                if outcome.success:
                    if goal_satisfied(goal, self.thor.observe()):
                        return AgentResult(True, goal, "Goal verified after action.", self.executor.total_steps)
                    continue

                failed = True
                recovery = self.recovery.classify(action, outcome, self.thor.observe())
                self.memory.add(
                    "recovery",
                    {
                        "reason": recovery.reason,
                        "failed_action": action.as_dict(),
                        "actions": [a.as_dict() for a in recovery.actions],
                    },
                )

                for recovery_action in recovery.actions[: self.recovery.max_retries_per_action]:
                    if self.executor.total_steps >= self.executor.max_steps:
                        break
                    recovery_outcome = self.executor.execute(recovery_action)
                    recent_context.append(
                        {
                            "recovery_action": recovery_action.as_dict(),
                            "success": recovery_outcome.success,
                            "error": recovery_outcome.error,
                        }
                    )
                    recent_context = recent_context[-12:]
                    if goal_satisfied(goal, self.thor.observe()):
                        return AgentResult(True, goal, "Goal verified during recovery.", self.executor.total_steps)

                break

            if self.executor.total_steps >= self.executor.max_steps:
                return AgentResult(False, goal, "Global primitive-action budget exhausted.", self.executor.total_steps)

            # Whether the plan succeeded partially or recovery ran, always observe and replan.
            if failed:
                continue

        return AgentResult(
            False,
            goal,
            "Replanning budget exhausted before the goal could be verified.",
            self.executor.total_steps,
        )
