from numpy import *
from linreg import *
from util import *


def myfunc(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / sqrt(x[1])
    return -(x[1]-1)*(x[1]-1) + 5


class optimize:
    def __init__(self, RBF_func = None, resampling = 0):
	# resampling specifies whether the length scale should be resampled
	# If resampling = 0 the no resampling is done
	# If resampling = n for n > 0 then resampling happens every n steps
	# NOTE: this is not finished
	
	self.num_basis = 500
	self.bounds = [(0.01,0.51), (10.0, 2000.0)]
	self.basis = lhsample(self.num_basis, self.bounds)
	self.v_0 = mat(eye(self.num_basis+1))
	self.w_0 = mat(zeros((self.num_basis+1, 1)))
	
	# NOTE: original value of a_0 and b_0 are 5 and 10.
	
	self.a_0 = 5.0
	self.b_0 = 10.0
	self.start_point = [0.1, 100]
	self.maxeval = 1000
	self.lb = []
	self.ub = []
	
	self.resampling = resampling
	
	self.RBF_func = RBF_func
	
	# NOTE: How to choose a_0 and b_0?
	# NOTE: This can be chosen in smart ways probably
	# NOTE: Look at inverse-gamma prior.
	self.epsilons =  arange(15.0, 20.0, 1.0)

	self.dim = len(self.bounds)
	self.linearmodel = group_linreg(self.epsilons, self.v_0, self.w_0, self.a_0, self.b_0, self.basis, RBF_func = self.RBF_func, resample = self.resampling)
	#self.objective_func = lambda x, grad, alpha: self.linearmodel.compute_UCB(x, alpha)
	self.objective_func = lambda x, grad, alpha: self.linearmodel.expected_improvement(x)
	self.set_lower_bound(); self.set_upper_bound()
	
	self.bf_opt_steps = [0.02, 100.0]
	
    def acquisition(self, x):
	return self.linearmodel.expected_improvement(x)
    
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
	self.v_0 = mat(eye(self.num_basis+1))*10e10
	self.w_0 = mat(zeros((self.num_basis+1, 1)))
	
	self.dim = len(self.bounds)
	self.set_lower_bound(); self.set_upper_bound()
	self.linearmodel = group_linreg(self.epsilons, self.v_0, self.w_0, self.a_0, self.b_0, self.basis, RBF_func = self.RBF_func)
	#self.objective_func = lambda x, grad, alpha: self.linearmodel.compute_UCB(x, alpha)
	self.objective_func = lambda x, grad, alpha: self.linearmodel.expected_improvement(x)
	
    def prob_obs_x_or_extm(self, x, y_n):
	return self.linearmodel.prob_obs_x_or_extm(x, y_n)

    def resample(self):
	self.linearmodel.re_sample()
	
    def update(self, x, y):
	# NOTE: Pay attention to the shape of x
	x = array(x)
	x = reshape(x, (1, shape(x)[0]))
	self.linearmodel.update(mat(x), y)

    def predict(self, x):
	x = array(x)
	#x = mat(reshape(x, (1, shape(x)[0])))
	
	mean,var = self.linearmodel.predict(x)
	
	return mean, var
	
    def bf_opt(self, alpha):
	"""
	Optimize by trying different values using brute force
	"""
	best_param = self.start_point
	
	# Set the objective function with alpha in consideration
	local_objective_func = lambda x, grad: self.objective_func(x, grad, alpha)
	
	best_objective = local_objective_func(best_param, None)
	range_list = []
	
	for i in range(len(self.bounds)):
	    tuples = self.bounds[i]
	    range_list.append(arange(tuples[0], tuples[1], self.bf_opt_steps[i]))

	choices = zeros((len(range_list), 1))
	best_param, best_objective = self.bf_opt_helper(range_list, 0, best_param, choices, best_objective, local_objective_func)
	
	return best_param
	    
	    
    def bf_opt_helper(self, range_list, i, best_param, choices, best_objective, local_objective_func):
	if i == len(range_list):
	    x = [range_list[j][int(choices[j])] for j in range(len(choices))]
	    objective = local_objective_func(x, 0)
	    if objective > best_objective:
		best_param = x
		best_objective = objective
		
		return best_param, best_objective
	elif i < len(range_list):
	    for j in range(len(range_list[i])):
		choices[i] = j
		best_param, best_objective = self.bf_opt_helper(range_list, i+1, best_param, choices, best_objective, local_objective_func)
	
	return best_param, best_objective
	
    def direct(self, alpha):
	import optimize_direct
	
	def obj_func(x):
	    return self.linearmodel.expected_improvement(x)

	optv, optx = optimize_direct.direct(obj_func, self.bounds, maxiter=self.maxeval)

	#print optv, optx

	## Using DIRECT as the optimization scheme
	#opt = nlopt.opt(nlopt.GN_DIRECT, self.dim)

	## Set the objective
	#opt.set_max_objective(fn)

	## Set the maximum number of iterations
	#opt.set_maxeval(self.maxeval)

	## Set lower and upper bounds
	#opt.set_lower_bounds(self.lb)
	#opt.set_upper_bounds(self.ub)

	## Optimize with starting point
	#x = opt.optimize(self.start_point)
	##minf = opt.last_optimum_value()
	##print "optimum at ", x[0]
	##print "minimum value = ", minf
	##print "result code = ", opt.last_optimize_result()

	return optx