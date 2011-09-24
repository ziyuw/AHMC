from optimize import *
import util
from numpy import *
from numpy.random import *

plot = True
if plot:
    from matplotlib.pyplot import *

def objective(seq):
    return float(0.125*log(seq)*cos(seq[0]/200.0)+1)

lambdas = array([4900])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

#opt.bounds = [(2.0, 7.0)]
#opt.num_basis = 50
#opt.start_point = [3.0]
#opt.maxeval = 100
#opt.epsilons =  arange(15.0, 16.0, 0.5)
#opt.bf_opt_steps = [0.1]
#opt.reinitialize()

#opt.bounds = [(105.0, 5005.0)]
#opt.num_basis = 100
#opt.start_point = [200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(13.5, 14.0, 0.5)
#opt.bf_opt_steps = [20.0]

opt.bounds = [(105.0, 5005.0)]
opt.num_basis = 50
opt.start_point = [150.0]
opt.maxeval = 100
opt.epsilons =  arange(6.5, 7.0, 0.5)
opt.bf_opt_steps = [20.0]
opt.reinitialize()

#opt.bounds = [(10.0, 1010.0)]
#opt.num_basis = 200
#opt.start_point = [50.0]
#opt.maxeval = 100
#opt.epsilons =  arange(13.0, 14.0, 0.5)
#opt.bf_opt_steps = [10.0]
#opt.reinitialize()

seq = arange(105.0, 5005.0, 1.0)
#y_value = -(0.5*seq*seq*seq - 0.1*seq*seq)
y_value = 0.125*log(seq)*cos(seq/200.0)+1

xs = []
ys = []
x = opt.start_point
for i in range(20):
    # x = opt.direct(float(i+1)) # Use direct to optimize
    
    noisy_y = objective(x) + normal(loc=0.0, scale=0.2)
    print i, x, noisy_y
    xs.append(x)
    ys.append(noisy_y)
    opt.update(x, noisy_y)
    x = opt.bf_opt(float(i+1)) # Use brute force to optimize
    #x = opt.direct(float(i+1))
    
print "Training finished."

# Testing
test_range = arange(105.0, 5005.0, 20.0)
predictions = []
upper = []
lower = []
for pt in test_range:
    mu, sigma = opt.predict(mat(pt))
    predictions.append(mu)
    upper.append(mu+sigma)
    lower.append(mu-sigma)

if plot:
    #print predictions
    line1 = plot(seq, y_value)
    line2 = plot(test_range, predictions, '-r*')
    line3 = plot(test_range, upper, '--')
    line4 = plot(test_range, lower, '-.k')
    line5 = plot(xs, ys, 'bo')

    figlegend( (line1, line2, line3, line4, line5),
	    ('True Function', 'Predictions (After Opt.)', 'Upper Confidence Bound', 'Lower Confidence Bound', 'Sampled points (During Opt.)'),
	    'lower right' )
    show()