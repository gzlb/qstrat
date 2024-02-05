from dataclasses import dataclass
from typing import Callable, List, Protocol

import numpy as np


@dataclass
class OrnsteinUhlenbeckParameters:
    alpha: float
    kappa: float
    sigma: float

@dataclass
class VasicekParameters:
    alpha: float
    beta: float
    sigma: float

class StochasticProcess(Protocol):
    @property
    def drift(self) -> Callable[[float], float]:
        ...

    @property
    def diffusion(self) -> Callable[[float], float]:
        ...

@dataclass
class OrnsteinUhlenbeck(StochasticProcess):
    parameters: OrnsteinUhlenbeckParameters

    @property
    def drift(self) -> Callable[[float], float]:
        return lambda St: self.parameters.alpha * (self.parameters.kappa - St)

    @property
    def diffusion(self) -> float:
        return self.parameters.sigma

@dataclass
class Vasicek(StochasticProcess):
    parameters: VasicekParameters

    @property
    def drift(self) -> Callable[[float], float]:
        return lambda St: self.parameters.alpha * (self.parameters.beta - St)

    @property
    def diffusion(self) -> Callable[[float], float]:
        return lambda St: self.parameters.sigma * np.sqrt(St)