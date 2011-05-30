from optimize import *
import util
from numpy import *
from numpy.random import *
from matplotlib.pyplot import *

def objective(x):
    return -float((x[0]-0.3)**2 + ((x[1]-450)/1450.0)**2) -float((x[0]-0.5)**2 + ((x[1]-1050)/1450.0)**2)

lambdas = array([0.5, 1450.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

#opt.bounds = [(0.2, 0.6), (50.0, 5000.0)]
#opt.num_basis = 200
#opt.start_point = [0.4, 300.0]
#opt.maxeval = 100
#opt.epsilons =  arange(2.5, 3.0, 0.5)
#opt.bf_opt_steps = [0.05, 100.0]

opt.bounds = [(0.2, 0.7), (50.0, 1500.0)]
opt.num_basis = 100
opt.start_point = [0.4, 200.0]
opt.maxeval = 100
opt.epsilons =  arange(1.5, 2.0, 0.5)
opt.bf_opt_steps = [0.05, 50.0]

opt.reinitialize()

x = opt.start_point
for i in range(100):
    noisy_y = objective(x) + normal(loc=0.0, scale=0.1)
    print i, x, noisy_y
    opt.update(x, noisy_y)
    x = opt.bf_opt(float(i+1)) # Use brute force to optimize
    # x = opt.direct(float(i+1)) # Use direct to optimize