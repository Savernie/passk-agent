from dataclasses import dataclass
from typing import Protocol, Any

@dataclass
class Task:
    id: str
    data: Any  # tau2's own Task object

@dataclass
class Score:
    passed: bool
    reward: float
    details: dict

class Benchmark(Protocol):
    def load(self, split: str) -> list[Task]: ...
    def build_env(self, task: Task) -> Any: ...
    def score(self, task: Task, simulation: Any) -> Score: ...