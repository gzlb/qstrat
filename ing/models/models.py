from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Model(ABC):
    def __init__(
        self, has_exact_density: bool = False, default_sim_method: str = "Euler"
    ):
        """
        Base 1D model for SDE, defined by

        dX(t) = mu(X,t)dt + sigma(X,t)dW_t

        :param has_exact_density: bool, set to true if an exact density is implemented
        :param default_sim_method: str, the default method for simulating. This can be overridden by the simulator,
            but this allows you to set a good default (or perhaps exact method) per model
        """
        self._has_exact_density = has_exact_density
        self._params: Optional[np.ndarray] = None
        self._positive = (
            False  # updated when params are set, indicates positivity of process
        )
        self._default_sim_method = default_sim_method

    @abstractmethod
    def drift(self, x: float, t: float) -> float:
        """The drift term of the model"""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x: float, t: float) -> float:
        """The diffusion term of the model"""
        raise NotImplementedError

    # ==============================
    # Optimization/Fit interface
    # ==============================

    @property
    def params(self) -> np.ndarray:
        """Access the parameters"""
        return self._params

    @params.setter
    def params(self, vals: np.ndarray):
        """Set parameters, used by fitter to move through param space"""
        self._positive = self._set_is_positive(
            params=vals
        )  # Check if the params ensure positive density
        self._params = vals

    @property
    def has_exact_density(self) -> bool:
        """Return true if the model has an exact density implemented"""
        return self._has_exact_density

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        """
        In the case where the exact transition density,
        P(Xt, t | X0) is known, override this method
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t0: float, the time of observing x0
        :param dt: float, the time step between x0 and xt
        :return: probability
        """
        raise NotImplementedError

    def exact_step(self, t: float, dt: float, x: float, dZ: float) -> float:
        """Exact Simulation Step, Implement if known (e.g., Brownian motion or GBM)"""
        raise NotImplementedError

    @property
    def is_positive(self) -> bool:
        """Check if the model has non-negative paths, given the currently set parameters"""
        return self._positive

    @property
    def default_sim_method(self) -> str:
        """Default method used for simulation"""
        return self._default_sim_method

    # ==============================
    # Private
    # ==============================

    def _set_is_positive(self, params: np.ndarray) -> bool:
        """
        Override this method to specify if the parameters ensure a non-negative process. This is used to
        ensure sample paths are positive. If this is not overridden, no protection is added to ensure positivity
        when simulating
        :param params: parameters, the positivity of the process can be parameter-dependent
        :return: bool, True if the parameters lead to a positive process
        """
        return False
