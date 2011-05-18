import nlopt
from numpy import *

def myfunc(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / sqrt(x[1])
    return -(x[1]-1)*(x[1]-1) + 5

# Using DIRECT as the optimization scheme
opt = nlopt.opt(nlopt.GN_DIRECT, 2)

# Set the objective
opt.set_max_objective(myfunc)

# Set the maximum number of iterations
opt.set_maxeval(1000)

# Set lower and upper bounds
opt.set_lower_bounds(-5.0)
opt.set_upper_bounds(5.0)

# Optimize with starting point
x = opt.optimize([1, 5])
minf = opt.last_optimum_value()
print "optimum at ", x[0],x[1]
print "minimum value = ", minf
print "result code = ", opt.last_optimize_result()