from dataclasses import dataclass
from typing import Union

import numpy as np

from ing.fit.models import Model
from abc import ABC, abstractmethod


@dataclass
class TransitionDensity(ABC):
    model: Model

    @abstractmethod
    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        raise NotImplementedError


@dataclass
class ExactDensity(TransitionDensity):
    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        return self.model.exact_density(x0=x0, xt=xt, t0=t0, dt=dt)
