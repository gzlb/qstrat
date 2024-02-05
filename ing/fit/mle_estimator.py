from typing import List, Tuple

import numpy as np

from ing.fit.ll_estimator import LikelihoodEstimator
from ing.fit.minimizer import Minimizer, ScipyMinimizer
from ing.fit.transition_density import TransitionDensity


class MLE(LikelihoodEstimator):
    def __init__(
        self,
        sample: np.ndarray,
        param_bounds: List[Tuple],
        dt: float,
        density: TransitionDensity,
        minimizer: Minimizer = ScipyMinimizer(),
        t0: float = 0,
    ):
        """
        Maximum likelihood estimator based on some analytical representation for the transition density.
        :param sample: np.ndarray, a single path draw from some theoretical model
        :param param_bounds: List[Tuple], one tuple (lower, upper) of bounds for each parameter
        :param dt: float, time step (time between diffusion steps)
            Either supply a constant dt for all time steps or supply a set of dt's equal in length to the sample
        :param density: TransitionDensity, transition density of some kind, attached to a model
        :param minimizer: Minimizer, the minimizer that is used to maximize the likelihood function. If none is
            supplied, then ScipyMinimizer is used by default
        :param t0: Union[float, np.ndarray], optional parameter, if you are working with a time-homogeneous model,
            then this doesn't matter. Else, it's the set of times at which to evaluate the drift and diffusion
             coefficients
        """
        super().__init__(
            sample=sample,
            param_bounds=param_bounds,
            dt=dt,
            model=density.model,
            minimizer=minimizer,
            t0=t0,
        )
        self._density = density

    def log_likelihood_negative(self, params: np.ndarray) -> float:
        self._model.params = params
        return -np.sum(
            np.log(
                np.maximum(
                    self._min_prob,
                    self._density(
                        x0=self._sample[:-1],
                        xt=self._sample[1:],
                        t0=self._t0,
                        dt=self._dt,
                    ),
                )
            )
        )
