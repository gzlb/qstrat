from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from ing.fit.ll_estimator import LikelihoodEstimator
from ing.fit.transition_density import TransitionDensity
from ing.fit.minimizer import Minimizer, ScipyMinimizer


@dataclass
class AnalyticalMLE(LikelihoodEstimator):
    sample: np.ndarray
    param_bounds: List[Tuple]
    dt: Union[float, np.ndarray]
    density: TransitionDensity
    minimizer: Minimizer = ScipyMinimizer()
    t0: Union[float, np.ndarray] = 0

    def log_likelihood_negative(self, params: np.ndarray) -> float:
        self.model.params = params
        return -np.sum(np.log(np.maximum(self._min_prob,
                                         self.density(x0=self.sample[:-1],
                                                      xt=self.sample[1:],
                                                      t0=self.t0,
                                                      dt=self.dt))))
