import importlib
from tau2.runner.helpers import get_tasks as tau2_get_tasks
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from tau2.orchestrator.modes import CommunicationMode
from benchmarks.base import Task, Score


class Tau2Benchmark:
    def __init__(self, domain: str):
        self.domain = domain
        self._env_module = importlib.import_module(f"tau2.domains.{domain}.environment")

    def load(self, split: str = "base") -> list[Task]:
        raw = tau2_get_tasks(task_set_name=self.domain, task_split_name=split)
        return [Task(id=t.id, data=t) for t in raw]

    def build_env(self, task: Task):
        env = self._env_module.get_environment()
        init = task.data.initial_state
        if init and init.initialization_actions:
            for action in init.initialization_actions:
                toolkit = env.tools if action.env_type == "assistant" else env.user_tools
                getattr(toolkit, action.func_name)(**action.arguments)
        return env

    def score(self, task: Task, simulation) -> Score:
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task.data,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self.domain,
            mode=CommunicationMode.HALF_DUPLEX,
        )
        return Score(
            passed=reward_info.reward == 1.0,
            reward=reward_info.reward,
            details=reward_info.model_dump(),
        )