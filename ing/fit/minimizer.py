from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

@dataclass
class Result:
    params: np.ndarray
    value: float
    success: bool
    message: str = ""

class Minimizer(ABC):
    @abstractmethod
    def minimize(self,
                 function: Callable,
                 bounds: List[Tuple] = None,
                 guess: np.ndarray = None) -> Result:
        raise NotImplementedError

@dataclass
class ScipyMinimizer(Minimizer):
    method: str = 'trust-constr'
    tol: float = 5e-03
    options: Optional[dict] = None

    def __post_init__(self):
        self._options = self.options or {'maxiter': 250, 'gtol': 1e-06, 'xtol': 1e-04, 'verbose': 1}

    def minimize(self,
                 function: Callable,
                 bounds: Optional[List[Tuple]] = None,
                 guess: Optional[np.ndarray] = None) -> Result:
        res = minimize(function,
                       guess,
                       tol=self.tol,
                       method=self.method,
                       bounds=bounds,
                       options=self._options)

        return Result(params=res.x, value=res.fun, success=res.success, message=res.message)
