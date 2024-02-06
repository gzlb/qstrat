from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ing.models.models import Model


class Scheme(ABC):
    def __init__(self, model: Model):
        """
        Base Simulation Stepper class, which is responsible for implementing a single step of a time-discretization
        scheme, e.g., Euler. Given the current state, it knows how to evolve the state by one time step, and is
        called sequentially during a path simulation.
        :param model: the SDE model
        """
        self._model = model

    @property
    def model(self) -> Model:
        """Access to the underlying model"""
        return self._model

    @abstractmethod
    def next(
        self,
        t: float,
        dt: float,
        x: Union[float, np.ndarray],
        dZ: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Given the current state and random variate(s), evolve state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state after evolving by one step
        """
        raise NotImplementedError

    def __call__(
        self,
        t: float,
        dt: float,
        x: Union[float, np.ndarray],
        dZ: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Same as a call to next()"""
        return self.next(t=t, dt=dt, x=x, dZ=dZ)

    @staticmethod
    def new_scheme(scheme: str, model: Model):
        """
        Factory method to construct a simulation stepper according to scheme
        :param scheme: str, name of the simulation scheme, e.g.
            'Euler', 'Milstein', 'Milstein2', 'Exact'
        :param model: Model1D, the SDE model to which the stepper is bound
        :return: Stepper, bound to the model, for a particular scheme
        """
        if scheme == "Euler":
            return EulerScheme(model=model)

        else:
            raise NotImplementedError


class ExactScheme(Scheme):
    def __init__(self, model: Model):
        """
        Exact Simulation Step
        :param model: the SDE model
        """
        super().__init__(model=model)

    def next(
        self,
        t: float,
        dt: float,
        x: Union[float, np.ndarray],
        dZ: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Given the current state and random variate(s), evolve state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state after evolving by one step
        """
        return self._model.exact_step(t=t, dt=dt, x=x, dZ=dZ)


class EulerScheme(Scheme):
    def __init__(self, model: Model):
        """
        Euler Simulation Step
        :param model: the SDE model
        """
        super().__init__(model=model)

    def next(
        self,
        t: float,
        dt: float,
        x: Union[float, np.ndarray],
        dZ: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Given the current state and random variate(s), evolve state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state after evolving by one step
        """
        xp = (
            x
            + self._model.drift(x, t) * dt
            + self._model.diffusion(x, t) * np.sqrt(dt) * dZ
        )
        return np.maximum(0.0, xp) if self._model.is_positive else xp
