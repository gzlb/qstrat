# ING 

## Description

This project consists of the assignment of fitting stochastic processes to a given spread data. The goal is to capture the mean-reversion rate and as such devise possible arbitrages. This simple strategy aims to capture profits as the spread reverts to its mean, while also managing risk through the use of stop-loss orders.  

The main overview of the method can be summarized as follows: first, the parameters of a stochastic process are estimated using Maximum Likelihood Estimation (MLE), which is then next used to fit and generate simulations by a Monte Carlo. 

The notebooks are an excellent starting point to consider. 

## Installation 

This project is managed using Poetry. The dependencies are listed in the pyproject.toml file, and they will be installed automatically when you run poetry install.

To use this project, follow these steps:

1. Clone the repository from GitHub:

```
git clone https://github.com/gzlb/ing.git
```

2. Navigate to the folder 
```
cd ing 
```

3. Install the project dependencies using Poetry:
```
poetry install
```

## Project Structure 

- data: Contains spreadsheets with spread data.
- dist: Contains distribution files.
- ing: Main package directory.
- fit: Module for fitting models to data using maximum likelihood estimation.
- models: Module for defining financial models (currently support Ornstein-Uhlenbeck and Cox-Ingersoll-Ross).
- optm: Module for optimization functions.
- plots: Module for plotting functions.
- sim: Module for simulating spread movements using the stochastic processes.
- utils: Module for data tools.
- notebooks: Jupyter notebooks for analysis and demonstration of proof of concept.

## Usage
This project provides functionalities to analyze spread data. The notebooks are an excellent starting point to consider. The main workflow can summarized with following () steps

1. Read the data, stored in the datafolder by using functionality in *data_utils.py* and strip it correctly to pipe it for the models
2. Initialize: 
    * The parameters of the optimization (guess and boundary)
    * The discretization of the stochastic process (timestep and horizon) 
    * The stochastic process (model) of choice (Ornstein-Uhlenbeck or CIR), by importing Ornstein-Uhlenbeck or CIR from respectively ou.py and cir.py 
    * The density which belongs to the model, by importing ExactDensity from transition_density.py 
3. Pipe the variables of steps 1 and 2 in to the MLE object to run the maximum likelihood estimation, by importing MLE from mle_estimator.py  
4. Run a simulation step using the simulator, by importing Simulator from simulator.py. Note that the simulation is performed with the function sim_path, which can also generate multiple paths, in case of using a Monte Carlo.   
5. Plot the results and optionally summary statistics by importing the tools of data_utils.py and or plot.py 


