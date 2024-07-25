import numpy as np
import pysindy as ps
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pysindy.differentiation import FiniteDifference

from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
from emukit.core.constraints import LinearInequalityConstraint
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

class BayOpt(object):
    """docstring for BayOpt"""
    def __init__(self, target_function, n_samples): #,parameter_space, constraint = None
        self.target_function = target_function
        self.n_samples = n_samples
        # self.parameter_space = parameter_space
        # self.constraint = costraint
    def run(self, max_iterations):
        parameter_space = ParameterSpace([DiscreteParameter('delay1', range(25,500,25)), DiscreteParameter('delay2', range(25,500,25))]) # al posto lista self.parameter_space
        constraint = LinearInequalityConstraint(np.array([[1,-1]]), lower_bound = np.array([1]), upper_bound = np.array([500])) #constraint = self.constraint
        parameter_space.constraints.append(constraint)
        #if self.constraint != None quello che c'è sopra
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
delay1s = np.zeros(num_of_it+1)
delay2s = np.zeros(num_of_it+1)

time_series = sio.loadmat('x1.mat')
x = np.array(time_series['x'][0])
time = sio.loadmat('time.mat')
t = np.array(time['tspan'][0])

# Prepare the data 
num = 500 # Number of snapshots used
n_snaps = num
adv = 500 # Time interval we are searching the delays in (from 0 to adv/100)
x_train = x[adv:adv+n_snaps]
t_train = t[0:n_snaps]
dt = t[1]-t[0]
der = np.gradient(x_train, dt)
norm_der = np.linalg.norm(der)

stlsq_optimizer = ps.STLSQ(threshold=0.1)

def SINDy_error(param):
    global count
    # print(f'count = {count}')
    delay1 = param[0][0]
    delay2 = param[0][1]
    if delay1 <= delay2:
        error = 1
    else:
        # print(f'delay = {del_val}')
    
        x_train_del1 = x[adv - delay1: adv-delay1+num]
        x_train_del2 = x[adv - delay2: adv-delay2+num]
        X=np.array([x_train, x_train_del1, x_train_del2]).T

        model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3), optimizer = stlsq_optimizer)
        model.fit(X, t_train)
        # model.print()
        pred = model.predict(X)

        error = np.linalg.norm(der - pred[:, 0])/norm_der
        err_vect[count] = error
        err_plot[count] = np.min(err_vect)
        delay1s[count] = int(delay1)
        delay2s[count] = int(delay2)
    print(count)
    count += 1

    return np.array(error).reshape(-1, 1)

# Bayesian optimization
i = 0


bopt = BayOpt(target_function=SINDy_error, n_samples=1)
results = bopt.run(max_iterations=num_of_it)
minimum = results.minimum_location
print(f'results.minimum_location = {minimum}')
print(f'results.minimum_value = {results.minimum_value}')
delay1_minimum = int(minimum[0])
delay2_minimum = int(minimum[1])
print(f'the parameters identified are = {minimum}')
iteration = np.where(err_plot == results.minimum_value)
print(iteration[0][0])
x_train_del1 = x[adv - delay1_minimum: adv - delay1_minimum + num]
x_train_del2 = x[adv - delay2_minimum: adv - delay2_minimum + num]
X = np.array([x_train, x_train_del1, x_train_del2]).T
model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3))
model.fit(X, t_train)
model.print()
