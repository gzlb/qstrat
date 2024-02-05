from abc import ABC, abstractmethod

from ing.models.models import Model


class TransitionDensity(ABC):
    def __init__(self, model: Model):
        """
        Class representing the transition density for a model, implementing a __call__ method to evaluate
        the transition density (bound to the model)

        :param model: the SDE model, referenced during calls to the transition density
        """
        self._model = model

    @property
    def model(self) -> Model:
        """Access to the underlying model"""
        return self._model

    @abstractmethod
    def __call__(self, x0: float, xt: float, t0: float, dt: float) -> float:
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be the same dimension as x0)
        :param t0: float, the time at which to evaluate the coefficients. Irrelevant for time-inhomogeneous models
        :param dt: float, the time step between x0 and xt
        :return: probability (the same dimension as x0 and xt)
        """
        raise NotImplementedError


class ExactDensity(TransitionDensity):
    def __init__(self, model: Model):
        """
        Class representing the exact transition density for a model (when available)

        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self, x0: float, xt: float, t0: float, dt: float) -> float:
        """
        The exact transition density (when applicable)
        Note: this will raise an exception if the model does not implement exact_density

        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be the same dimension as x0)
        :param t0: float, the time at which to evaluate the coefficients. Irrelevant for time-inhomogeneous models
        :param dt: float, the time step between x0 and xt
        :return: probability (the same dimension as x0 and xt)
        """
        return self._model.exact_density(x0=x0, xt=xt, t0=t0, dt=dt)
