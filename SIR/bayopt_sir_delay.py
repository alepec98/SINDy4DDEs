import numpy as np
import pysindy as ps
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pysindy.differentiation import FiniteDifference

from scipy.integrate import odeint

stlsq_optimizer = ps.STLSQ(threshold=0.5)

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
        parameter_space = ParameterSpace([DiscreteParameter('delay', range(25,425,1))])
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


num = 500
        
#os.chdir('/home/alessandro/Scrivania/Tirocinio/tirocinio_sindy/Articolo/Codice/sir_one_delay')
#os.chdir('/home/alessandro/Scrivania')
time_series = sio.loadmat('x.mat')
x = np.array(time_series['x'])
time = sio.loadmat('tsir.mat')
t = np.array(time['tspan'][0])
ttrain = t[0:num]
s = x[0]
i = x[1]
r = x[2]

dt = t[1]-t[0]

num_of_it= 200
count = 0
err_plot = np.ones(num_of_it+1)
err_vect = np.ones(num_of_it+1)
delays = np.zeros(num_of_it+1)


def SINDy_error(delay):
    global count
    error_array = []
    size = len(delay)
    print(delay,num, size)
    for d in range(size):
        del_val = delay[d][0]
        print(del_val)
        strain = s[del_val:del_val+num]
        itrain = i[del_val:del_val+num]
        itrain_del = i[0:num]
        
        s_der = np.gradient(strain, dt)
        i_der = np.gradient(itrain, dt)
        s_norm = np.linalg.norm(s_der)
        i_norm = np.linalg.norm(i_der)
        
        model = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=2), optimizer=stlsq_optimizer)

        X=np.array([strain, itrain, itrain_del]).T
        model.fit(X, ttrain)
        #model.print()
        pred = model.predict(X)
    
        s_pred = pred[:,0]
        i_pred = pred[:,1]    
        comp_error_s = s_der-s_pred
        comp_error_i = i_der-i_pred
        error_s = np.linalg.norm(comp_error_s/s_norm)
        error_i = np.linalg.norm(comp_error_i/i_norm)
        error = max(error_s, error_i)
        print(error)
        error_array.append([error])
        
        err_vect[count] = error
        err_plot[count] = np.min(err_vect)
        delays[count] = int(del_val)
        count = count + 1
        
    error_array = np.array(error_array)
    print(error_array)
    return error_array

# Bayesian optimization

bopt = BayOpt(target_function = SINDy_error, n_samples=1)
results = bopt.run(max_iterations=num_of_it)

minimum = int(results.minimum_location)
print(f'results.minimum_location = {minimum}')
print(f'results.minimum_value = {results.minimum_value}')
print(f'the delay identified is = {minimum}')
iteration = np.where(err_plot == results.minimum_value)
print(iteration[0][0])

strain = s[minimum:minimum+num]
itrain = i[minimum:minimum+num]
itrain_del = i[0:num]

model = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=2), optimizer=stlsq_optimizer)
X=np.array([strain, itrain, itrain_del]).T
model.fit(X, ttrain)
model.print()
pred = model.predict(X)
coefficients = model.coefficients()
print(coefficients)
equations = model.equations()
print(equations)

plt.plot(err_plot)
plt.title('Error plot during Bayesian optimization - SIR with one delay')
plt.yscale("log")
plt.ylabel("Error")
plt.xlabel("Number of iterarions")
plt.show()

