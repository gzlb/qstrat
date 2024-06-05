from typing import Union

import numpy as np
from scipy.special import ive

from qstrat.models.models import Model


class CIR(Model):
    """
    Model for CIR (Cox-Ingersoll-Ross)
    Parameters: [kappa, mu, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa * (mu - X)
        sigma(X,t) = sigma * sqrt(X)         (sigma > 0)
    """

    def __init__(self):
        super().__init__(has_exact_density=True)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (self._params[1] - x)

    def diffusion(
        self, x: Union[float, np.ndarray], t: float
    ) -> Union[float, np.ndarray]:
        return self._params[2] * np.sqrt(x)

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        kappa, mu, sigma = self._params
        theta1, theta2, theta3 = kappa * mu, kappa, sigma

        et = np.exp(-theta2 * dt)
        c = 2 * theta2 / (theta3**2 * (1 - et))
        u, v = c * x0 * et, c * xt
        q = 2 * theta1 / theta3**2 - 1

        z = 2 * np.sqrt(u * v)
        p = c * np.exp(-(u + v) + np.abs(z)) * (v / u) ** (q / 2) * ive(q, z)

        return p

    def _set_is_positive(self, params: np.ndarray) -> bool:
        """CIR is always non-negative"""
        return True
