# Remember: here at the end to obtain the delay you will have to do adv - the results of the optimization

import numpy as np
import pysindy as ps
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pysindy.differentiation import FiniteDifference

from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

class BayOpt(object):
    """docstring for BayOpt"""
    def __init__(self, target_function, n_samples):
        self.target_function = target_function
        self.n_samples = n_samples
        
    def run(self, max_iterations):
        parameter_space = ParameterSpace([DiscreteParameter('delay', range(25,476,1))])
        design = RandomDesign(parameter_space)
        inputs = design.get_samples(self.n_samples)
        outputs = self.target_function(inputs)
        model_gpy = GPRegression(inputs, outputs)
        model_emukit = GPyModelWrapper(model_gpy)
        
        expected_improvement = ExpectedImprovement(model=model_emukit)
        
        bayesopt_loop = BayesianOptimizationLoop(model=model_emukit,
                                                 space=parameter_space,
                                                 acquisition=expected_improvement,
                                                 batch_size=1)
        
        bayesopt_loop.run_loop(self.target_function, max_iterations)
        return bayesopt_loop.get_results()

# Number of iterations
num_of_it = 300

# Define everything needed for the plot of the error during iterations
count = 0
err_plot = np.ones(num_of_it+1)
err_vect = np.ones(num_of_it+1)
delays = np.zeros(num_of_it+1)


time_series = sio.loadmat('mg.mat')
x = np.array(time_series['x'][0])
time = sio.loadmat('time.mat')
t = np.array(time['tspan'][0])

# Prepare the data 
n_snaps = 1200 # Number of snapshots used
adv = 500 # Time interval we are searching the delays in (from 0 to adv/100)
x_train = x[adv:adv+n_snaps]
t_train = t[:n_snaps]
dt = t[1]-t[0]
der = np.gradient(x_train, dt)
norm_der = np.linalg.norm(der)

# Prepare SINDy optimizer
threshold_value = 0.4
stlsq_optimizer = ps.STLSQ(threshold=threshold_value, alpha=0.1, max_iter=200, fit_intercept=False)


def SINDy_error(delay):
    global count
    print(f'count = {count}')
    del_val = delay[0][0]
    # print(f'delay = {del_val}')
    
    x_train_delay = x[del_val:del_val+n_snaps]
    den = 1/(1+x_train_delay**10)
    X=np.array([x_train, x_train_delay, den]).T
    
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2), optimizer = stlsq_optimizer)
    model.fit(X, t_train)
    # model.print()
    pred = model.predict(X)
    
    error = np.linalg.norm(der - pred[:, 0])/norm_der
    err_vect[count] = error
    err_plot[count] = np.min(err_vect)
    delays[count] = int(del_val)
    count += 1

    return np.array(error).reshape(-1, 1)

# Bayesian optimization
bopt = BayOpt(target_function = SINDy_error, n_samples=1)
results = bopt.run(max_iterations=num_of_it)

minimum = int(results.minimum_location[0])
print(f'results.minimum_location = {minimum}')
print(f'results.minimum_value = {results.minimum_value}')
delay = 500 - minimum
print(f'the delay identified is = {delay}')
iteration = np.where(err_plot == results.minimum_value)
print(iteration[0][0])


x_train_delay = x[minimum:minimum+n_snaps]
den = 1/(1+x_train_delay**10)
X = np.array([x_train, x_train_delay, den]).T

model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2), optimizer=stlsq_optimizer)
model.fit(X, t_train)
pred = model.predict(X)
model.print()
coefficients = model.coefficients()
print(coefficients)
equations = model.equations()
print(equations)
