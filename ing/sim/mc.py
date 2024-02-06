from typing import Union

import numpy as np

from ing.models.models import Model
from ing.sim.scheme import Stepper


def monte_carlo_simulation(
    S0: float,
    M: int,
    dt: float,
    model: Model,
    sub_step: int = 5,
    seed: int = None,
    method: Union[str, Stepper] = "Default",
    num_simulations: int = 1000,
) -> np.ndarray:
    """
    Function for performing Monte Carlo simulations of diffusion (SDE) process.

    :param S0: float, initial value of process
    :param M: int, number of time steps (path will be size M+1, as it contains S0)
    :param dt: float, time step size
    :param model: obj, the model
    :param sub_step: int, (optional, default=1). If greater than 1, do multiple sub-steps on each dt interval to
        reduce bias.
    :param seed: int, the random seed (used for reproducibility of experiments)
    :param method: str, the simulation scheme to use only Euler
        If set to "Default", uses the default simulation defined by the model (for example, "Exact" if it is known)
        Also allows you to supply your own stepper if desired
    :param num_simulations: int, number of Monte Carlo simulations to perform
    :return: array, simulated paths of process (size M+1, num_simulations)
    """
    np.random.seed(seed=seed)

    def init_path():
        path = np.zeros((M + 1, num_simulations))
        path[0, :] = S0
        return path

    def make_stepper():
        if isinstance(method, str):
            current_method = model.default_sim_method if method == "Default" else method
            return Stepper.new_stepper(scheme=current_method, model=model)
        elif isinstance(method, Stepper):
            return method
        else:
            raise ValueError("Unsupported stepper method")

    def sim_substep():
        path = init_path()
        norms = np.random.normal(
            loc=0.0, scale=1.0, size=(M * sub_step, num_simulations)
        )
        dt_sub = dt / sub_step

        for i in range(M * sub_step):
            path[i + 1, :] = stepper.next(
                t=i * dt_sub, dt=dt_sub, x=path[i, :], dZ=norms[i, :]
            )

        return path[::sub_step]

    all_paths = np.zeros((M + 1, num_simulations))
    stepper = make_stepper()

    for i in range(num_simulations):
        if sub_step > 1 and method != "Exact":
            path = sim_substep()
        else:
            path = init_path()
            norms = np.random.normal(loc=0.0, scale=1.0, size=(M, num_simulations))
            for j in range(M):
                path[j + 1, :] = stepper(t=j * dt, dt=dt, x=path[j, :], dZ=norms[j, :])

        all_paths[:, i] = path[:, 0]

    return all_paths
