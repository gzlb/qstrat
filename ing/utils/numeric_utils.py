from typing import List 
import numpy as np 
from models.models import StochasticProcess


def euler_maruyama(process: StochasticProcess, initial_value: float, dt: float, num_steps: int) -> List[float]:
    path = [0.0] * (num_steps + 1)
    path[0] = initial_value

    for i in range(num_steps):
        dWt = np.random.normal(0, np.sqrt(dt))
        path[i + 1] = path[i] + process.drift(path[i]) * dt + process.diffusion(path[i]) * dWt

    return path


def conditional_expectation(process: StochasticProcess, start_value: float, target_level: float, T: float, dt: float, num_paths: int):
    terminal_values = []

    for _ in range(num_paths):
        path = euler_maruyama(process, start_value, dt, int(T/dt))
        terminal_values.append(path[-1])

    conditional_expectation = np.mean([val for val in terminal_values if val >= target_level])

    return conditional_expectation


def expected_time_to_hit(process: StochasticProcess, start_value: float, target_level: float, num_paths: int):
    hitting_times = []

    for _ in range(num_paths):
        path = [start_value]
        time_to_hit = 0

        while path[-1] < target_level:
            dWt = np.random.normal(0, 1)
            path.append(path[-1] + process.drift(path[-1]) + process.diffusion(path[-1]) * dWt)
            time_to_hit += 1

        hitting_times.append(time_to_hit)

    return hitting_times
