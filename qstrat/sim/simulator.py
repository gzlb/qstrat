from typing import Union

import numpy as np

from qstrat.models.models import Model
from qstrat.sim.scheme import Scheme


class SimulationSDE(object):
    def __init__(
        self,
        S0: float,
        M: int,
        dt: float,
        model: Model,
        sub_step: int = 5,
        seed: int = None,
        method: Union[str, Scheme] = "Default",
    ):
        """
        Class for simulating paths of diffusion (SDE) process
        Override the sim_path method

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
        """
        self._S0 = S0
        self._M = M
        self._dt = dt
        self._model = model
        self._method = method
        self._stepper = self._make_stepper(method, model)
        self._sub_step = sub_step

        self.set_seed(seed=seed)

    def set_seed(self, seed: int = None):
        np.random.seed(seed=seed)
        return self

    @property
    def model(self) -> Model:
        """Access the underlying model"""
        return self._model

    def sim_path(self, num_paths: int = 1) -> np.ndarray:
        """
        Simulate a new path(s) of size M + 1
        :param num_paths: int, number of independent paths to simulate. By default, only
            one path (column array) is returned. If num_paths > 1, each column is a path
        :return: array, path(s) of process
        """
        if self._sub_step > 1 and self._method != "Exact":
            return self._sim_substep(num_paths=num_paths)

        path = self._init_path(path_shape=(self._M + 1, num_paths))
        norms = np.random.normal(loc=0.0, scale=1.0, size=(self._M, num_paths))
        for i in range(self._M):
            path[i + 1, :] = self._stepper(
                t=i * self._dt, dt=self._dt, x=path[i, :], dZ=norms[i, :]
            )
        path = path.flatten()
        return path

    # ====================
    # PRIVATE
    # ====================

    def _init_path(self, path_shape: tuple):
        path = np.zeros(shape=path_shape)
        path[0, :] = self._S0
        return path

    def _make_stepper(self, method: Union[str, Scheme], model: Model) -> Scheme:
        if isinstance(method, str):
            self._method = model.default_sim_method if method == "Default" else method
            return Scheme.new_scheme(scheme=self._method, model=self.model)
        elif isinstance(method, Scheme):
            self._method = "custom"
            return method
        else:
            raise ValueError("Unsupported stepper method")

    def _sim_substep(self, num_paths: int) -> np.ndarray:
        """simulate using the sub-stepping routine (reduced bias)"""
        path = self._init_path(path_shape=(self._M * self._sub_step + 1, num_paths))
        norms = np.random.normal(
            loc=0.0, scale=1.0, size=(self._M * self._sub_step, num_paths)
        )
        dt_sub = (
            self._dt / self._sub_step
        )  # divides dt into subintervals of length dt_sub

        for i in range(self._M * self._sub_step):
            path[i + 1, :] = self._stepper.next(
                t=i * dt_sub, dt=dt_sub, x=path[i, :], dZ=norms[i, :]
            )

        return path[:: self._sub_step]
