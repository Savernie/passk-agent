from typing import Protocol, Any


class Task:
    id: str
    data: dict  # whatever the benchmark's raw task definition looks like


class Env:
    """Whatever 'the world' looks like for this benchmark —
    a simulated airline DB for τ²-bench, a document corpus for BrowseComp-Plus."""
    ...


class Score:
    passed: bool
    reward: float
    details: dict


class Benchmark(Protocol):
    def load(self, split: str) -> list[Task]:
        """Return the tasks to run, e.g. 'base' split, 10 airline tasks."""
        ...

    def build_env(self, task: Task) -> Env:
        """Set up a fresh world for this one task —
        e.g. a clean simulated airline DB in the right starting state."""
        ...

    def score(self, task: Task, trajectory: list[dict]) -> Score:
        """After the agent+user conversation finished, judge it —
        e.g. hash-compare DB state, check required phrases."""
        ...