from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
from typing import List, Callable 

from ing.fit.minimizer import Minimizer, ScipyMinimizer
from ing.fit.estimator import Estimator, EstimatedResult

@dataclass
class LikelihoodEstimator(Estimator):
    min_prob: float = 1e-30
    minimizer: Minimizer = field(default=ScipyMinimizer())

    def estimate_params(self, params0: np.ndarray) -> EstimatedResult:
        return self._estimate_params(params0=params0, likelihood=self.log_likelihood_negative)

    @abstractmethod
    def log_likelihood_negative(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def _estimate_params(self, params0: np.ndarray, likelihood: Callable) -> EstimatedResult:
        print(f"Initial Params: {params0}")
        print(f"Initial Likelihood: {-likelihood(params0)}")

        res = self.minimizer.minimize(function=likelihood, bounds=self.param_bounds, guess=params0)
        params = res.params

        final_like = -res.value
        print(f"Final Params: {params}")
        print(f"Final Likelihood: {final_like}")
        return EstimatedResult(params=params, log_like=final_like, sample_size=len(self.sample) - 1)
