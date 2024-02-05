from scipy.optimize import minimize
import numpy as np
from typing import Tuple 
from dataclasses import dataclass
from typing import Callable, List, Protocol, Type 
from ing.models.models import StochasticProcess

def ornstein_uhlenbeck_mle_estimate(x:np.array) -> Tuple[float, float, float]:
    """
    Reads a numpy array and returns the Ornstein Uhlenbeck parameter estimation using MLE

    Parameters:
    - x numpy array (np.array)
    
    Returns: 
    - estimation of all three parameters (tuple) 
    """
    s_x = np.sum(x[:-1])
    s_y = np.sum(x[1:])
    s_xx = np.sum(x[:-1]**2)
    s_yy = np.sum(x[1:]**2)
    s_xy = np.sum(x[:-1] * x[1:])
    n = len(x)-1
    delta = 1

    kappa = ((s_y*s_xx)-(s_x*s_xy))/(n*(s_xx-s_xy)-((s_x**2)-s_x*s_y)) # Mean

    alpha = -(1/delta)*np.log((s_xy-kappa*s_x-kappa*s_y+n*kappa**2)/(s_xx-2*kappa*s_x+n*kappa**2)) # Rate

    beta = np.exp(-alpha*delta)
    sigma_h = np.sqrt((1/n)*(s_yy-(2*beta*s_xy)+((beta**2)*s_xx)-(2*kappa*(1-beta)*(s_y-beta*s_x))+(n*(kappa**2)*(1-beta)**2)))

    sigma = np.sqrt((sigma_h**2)*(2*alpha/(1-beta**2))) 

    return alpha, kappa, sigma 


from scipy.optimize import minimize

@dataclass
class MLESolver:
    process_model: Type[StochasticProcess]

    def fit(self, x: np.ndarray) -> Tuple:
        # Define the negative log-likelihood function for the given process model
        def neg_log_likelihood(parameters):
            process = self.process_model(parameters)
            drift = process.drift
            diffusion = process.diffusion

            # Compute the log-likelihood for the dataset
            log_likelihood = 0.5 * np.sum((x[1:] - drift(x[:-1]))**2 / diffusion(x[:-1])**2
                                           + np.log(diffusion(x[:-1])**2))
            return log_likelihood

        # Initial guess for parameters
        initial_parameters = np.ones(len(self.process_model.__dataclass_fields__))

        # Perform the MLE optimization
        result = minimize(neg_log_likelihood, initial_parameters, method='L-BFGS-B')

        # Extract the optimized parameters
        optimized_parameters = result.x

        return tuple(optimized_parameters)

