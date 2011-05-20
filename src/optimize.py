import nlopt
from numpy import *
from linreg import *
from util import *
from matplotlib.pyplot import *


def myfunc(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / sqrt(x[1])
    return -(x[1]-1)*(x[1]-1) + 5


class optimize:
    def __init__(self, RBF_func = None):
	self.num_basis = 200
	self.bounds = [(0.01,0.2), (5.0, 2000.0)]
	self.basis = lhsample(self.num_basis, self.bounds)
	self.v_0 = mat(eye(self.num_basis))
	self.w_0 = mat(zeros((self.num_basis, 1)))
	
	self.a_0 = 0.1
	self.b_0 = 0.1
	self.start_point = [0.1, 100]
	self.maxeval = 1000
	self.lb = []
	self.ub = []
	
	self.RBF_func = RBF_func
	
	# NOTE: How to choose a_0 and b_0?
	# NOTE: This can be chosen in smart ways probably
	self.epsilons =  arange(15.0, 20.0, 1.0)

	self.dim = len(self.bounds)
	self.linearmodel = group_linreg(self.epsilons, self.v_0, self.w_0, self.a_0, self.b_0, self.basis, RBF_func = self.RBF_func)
	self.set_lower_bound(); self.set_upper_bound()
	
    def set_lower_bound(self):
	self.lb = []
	for tuples in self.bounds:
	    self.lb.append(tuples[0])

    def set_upper_bound(self):
	self.ub = []
	for tuples in self.bounds:
	    self.ub.append(tuples[1])

    def reinitialize(self):
	self.basis = lhsample(self.num_basis, self.bounds)
	self.v_0 = mat(eye(self.num_basis))
	self.w_0 = mat(zeros((self.num_basis, 1)))
	
	self.dim = len(self.bounds)
	self.set_lower_bound(); self.set_upper_bound()
	self.linearmodel = group_linreg(self.epsilons, self.v_0, self.w_0, self.a_0, self.b_0, self.basis, RBF_func = self.RBF_func)
	
    def update(self, x, y):
	# NOTE: Pay attention to the shape of x
	x = array(x)
	x = reshape(x, (1, shape(x)[0]))
	self.linearmodel.update(mat(x), y)

    def predict(self, x):
	x = array(x)
	x = mat(reshape(x, (1, shape(x)[0])))
	
	mean,var = self.linearmodel.predict(x)
	
	return mean, var
	
    def direct(self, alpha):
	fn = lambda x, grad: self.linearmodel.compute_UCB(x, alpha)
	
	# Using DIRECT as the optimization scheme
	opt = nlopt.opt(nlopt.GN_DIRECT, self.dim)

	# Set the objective
	opt.set_max_objective(fn)

	# Set the maximum number of iterations
	opt.set_maxeval(self.maxeval)

	# Set lower and upper bounds
	opt.set_lower_bounds(self.lb)
	opt.set_upper_bounds(self.ub)

	# Optimize with starting point
	x = opt.optimize(self.start_point)
	#minf = opt.last_optimum_value()
	#print "optimum at ", x[0]
	#print "minimum value = ", minf
	#print "result code = ", opt.last_optimize_result()

	return x