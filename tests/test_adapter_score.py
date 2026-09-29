from pathlib import Path
from tau2.data_model.simulation import Results
from benchmarks.base import Task as OurTask
from benchmarks.tau2_adapter import Tau2Benchmark

RESULTS_PATH = Path("../tau2-bench/data/simulations/20260918_195254_airline_llm_agent_deepseek-chat_user_simulator_deepseek-chat/results.json")

results = Results.model_validate_json(RESULTS_PATH.read_text())

task = results.tasks[0]
simulation = results.simulations[0]

bench = Tau2Benchmark(domain="airline")
our_task = OurTask(id=task.id, data=task)
our_score = bench.score(our_task, simulation)

original_reward = simulation.reward_info.reward if simulation.reward_info else None

print(f"Original reward (from tau2 run): {original_reward}")
print(f"Our adapter's reward:            {our_score.reward}")
print(f"Match: {original_reward == our_score.reward}")