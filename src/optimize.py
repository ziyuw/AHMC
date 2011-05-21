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
	self.objective_func = lambda x, grad: self.linearmodel.compute_UCB(x, alpha)
	self.set_lower_bound(); self.set_upper_bound()
	
	self.bf_opt_steps = [0.01, 10.0]
	
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
	self.objective_func = lambda x, grad: self.linearmodel.compute_UCB(x, alpha)
	
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
	
    def bf_opt(self):
	"""
	Optimize by trying different values using brute force
	"""
	best_param = self.start_point
	best_objective = self.objective_func(best_param, 0)
	range_list = []
	
	for i in range(len(self.bounds)):
	    tuples = self.bounds[i]
	    range_list.append(arange(tuples[0], tuples[1], self.bf_opt_steps[i]))

	best_param, best_objective = self.bf_opt_helper(range_list, 0, best_param, choices, best_objective)
	
	return best_param
	    
	    
    def bf_opt_helper(self, range_list, i, best_param, choices, best_objective):
	if i == len(range_list):
	    x = [range_list[i][choices[i]] for i in len(choices)]
	    objective = self.linearmodel.compute_UCB(x, alpha)
	    if objective > best_objective:
		best_param = x
		best_objective = self.objective_func(x, grad)
		
		return best_param, best_objective
	elif i < len(range_list):
	    for j in range(len(range_list[i])):
		choices[i] = j
		best_param, best_objective = self.bf_opt_helper(range_list, i+1, best_param, choices, best_objective)
	
	return best_param, best_objective
	
    def direct(self, alpha):
	fn = self.objective_func
	
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