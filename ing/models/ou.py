from typing import Union

import numpy as np
from scipy.stats import norm

from ing.models.models import Model


class OrnsteinUhlenbeck(Model):
    """
    Model for OU (Ornstein-Uhlenbeck):
    Parameters: [kappa, mu, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = alpha * (kappa - X)
        sigma(X,t) = sigma * X
    """

    def __init__(self):
        super().__init__(has_exact_density=True)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (self._params[1] - x)

    def diffusion(
        self, x: Union[float, np.ndarray], t: float
    ) -> Union[float, np.ndarray]:
        return self._params[2] * (x > -10000)

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        alpha, kappa, sigma = self._params
        mu = kappa + (x0 - kappa) * np.exp(-alpha * dt)
        var = (1 - np.exp(-2 * alpha * dt)) * (sigma * sigma / (2 * alpha))
        return norm.pdf(xt, loc=mu, scale=np.sqrt(var))
