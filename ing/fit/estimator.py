from dataclasses import dataclass, property
from abc import ABC
import numpy as np
from typing import List, Tuple, Union

from ing.fit.models import Model

@dataclass
class EstimatedResult:
    params: np.ndarray
    log_like: float
    sample_size: int

    @property
    def likelihood(self) -> float:
        """ The likelihood with estimated params """
        return np.exp(self.log_like)

    @property
    def aic(self) -> float:
        """ The AIC (Aikake Information Criteria) with estimated params """
        return 2 * (len(self.params) - self.log_like)

    @property
    def bic(self) -> float:
        """ The BIC (Bayesian Information Criteria) with estimated params """
        return len(self.params) * np.log(self.sample_size) - 2 * self.log_like

    def __str__(self):
        """ String representation of the class (for pretty printing the results) """
        return f'\nparams      | {self.params} \n' \
               f'sample size | {self.sample_size} \n' \
               f'likelihood  | {self.log_like} \n' \
               f'AIC         | {self.aic}\n' \
               f'BIC         | {self.bic}'

@dataclass
class Estimator(ABC):
    sample: np.ndarray
    dt: Union[float, np.ndarray]
    model: Model
    param_bounds: List[Tuple]
    t0: Union[float, np.ndarray] = 0

    def __post_init__(self):
        if isinstance(self.dt, np.ndarray):
            if len(self.dt) != len(self.sample) - 1:
                raise ValueError("If you supply a sequence of dt, it must be the same size as the sample - 1")
            if len(self.dt.shape) != len(self.sample.shape):
                raise ValueError("The second dimension of the dt and sample vectors must agree, should be 1")

        if isinstance(self.t0, np.ndarray):
            if len(self.t0) != len(self.sample) - 1:
                raise ValueError("If you supply a sequence of t0, it must be the same size as the sample - 1")
            if len(self.t0.shape) != len(self.sample.shape):
                raise ValueError("The second dimension of the t0 and sample vectors must agree, should be 1")

    def estimate_params(self, params0: np.ndarray):
        """
        Main estimation function
        :param params0: array, the initial guess params
        :return: result, the estimated params and final likelihood
        """
        raise NotImplementedError
